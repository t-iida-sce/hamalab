
# 実装参考に使用
# https://github.com/huggingface/pytorch-pretrained-BERT

# Copyright (c) 2018 Hugging Face
# Released under the Apache License 2.0
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/LICENSE


# 必要なパッケージのimport
import os
import yaml
import copy
import math
import collections
import numpy as np

import torch
from torch import nn
import torchvision.models as CNNmodels

from models.resnet_base_network import Net18 # byol
from models.ae_modules import ResNet_AE, ResNet_VAE
from models.mlp_head import MLP_Classification, MLP_Classification3

# LayerNormalization層を定義
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization層です。
        学習済みモデルをそのままロードするため、学習済みモデルの変数名に変えています。
        オリジナルのGitHubの実装から変数名を変えています。
        weight→gamma、bias→beta
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# TE(localTE)のEmbeddingsモジュール
class Embeddings(nn.Module):
    """文章の単語ID列と、1文目か2文目かの情報を、埋め込みベクトルに変換する
    """

    def __init__(self, config):
        super(Embeddings, self).__init__()

        # 2つのベクトル表現の埋め込み
        self.hidden_size = config.hidden_size
        self.num_att_head = config.num_attention_heads
        self.imagesize = config.input_size
        self.patch_num_v = config.patch_num_v
        self.patch_num_h = config.patch_num_h
        self.num_lead = config.num_lead
        self.cls_tokens = nn.Parameter(torch.zeros(1, 1 + self.num_lead, config.hidden_size)) # cls_dx, cls_lead を作成
        
        # 1. 誘導画像をパッチに分割
        # 2. パッチごとにTransformer Encoderに入力する
        # 3. 得られた特徴量を平坦化(or nnにより抽出)し, それを各誘導から得られた特徴量とする (Emdbedding)

        # Transformer Positional Embedding：位置情報テンソルをベクトルに変換
        # Transformerの場合はsin、cosからなる固定値だったが、BERTは学習させる

        # 画像パッチに線形変換する
        patch_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h)) # 画像における列方向に分割, 分割された数がembeddingする数となる
        
        if config.overlap:
            stride_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h/2)) # 画像における列方向に分割, 分割された数がembeddingする数となる
            self.patch_embeddings = nn.Conv2d(in_channels=3, out_channels=config.hidden_size,kernel_size=patch_size,stride=stride_size,bias=False)
            
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html 出力されるtoken数を求める
            self.max_position_embedding = self.num_lead * int((self.imagesize[1] - (patch_size[1]-1) -1 )/stride_size[1]+1) + self.num_lead+1
        else:
            self.patch_embeddings = nn.Conv2d(in_channels=3, out_channels=config.hidden_size,kernel_size=patch_size,stride=patch_size,bias=False)
            self.max_position_embedding = self.num_lead*self.patch_num_v*self.patch_num_h+self.num_lead+1
                                                           
        self.position_embeddings = nn.Embedding(self.max_position_embedding, config.hidden_size)
                                                                        
        # zero tensor に対して畳み込みを行った場合, biasの影響で0でない値ができてしまう。そのため, 画像が存在しない場合はそのままzerotensorを作成するよう修正が必要
        #nn.init.normal_(self.patch_embeddings.bias, 0)

        # 作成したLayerNormalization層
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_leads, attention_show_flg=False):
        '''
        input_leads： [batch_size, number of lead, C, H, W]の文章の各誘導画像のテンソル
        '''

        # 1. Token Embeddings 
        batch_size = input_leads.size(0)
        input_leads = input_leads.permute(0,2,1,3,4) # (batch_size, number of lead, C, H, W) ⇒ (batch_size, C, number of lead,H, W )
        input_leads = input_leads.reshape(batch_size, input_leads.size(1), -1, input_leads.size(4)) #(batch_size, C, number of lead*H, W)
        
        # 診断用のcls_dx , 誘導用のcls_leadを用意する
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)

        x = self.patch_embeddings(input_leads) # ViTに倣い畳み込みでパッチを抽出
        x = x.flatten(2) # 2次元に変換
        x = x.transpose(-1, -2) 
        x = torch.cat((cls_tokens, x), dim=1)

        # 2. Transformer Positional Embedding：
        # [0, 1, 2 ... 誘導数 + 1(cls)]と誘導数+cls分の数字が1つずつ昇順に入った
        # [batch_size, self.num_lead*self.patch_num+1] の テンソルposition_leadsを作成
        # position_leadsを入力して、position_embeddings層から分散表現の次元数のテンソルを取り出す

        # 誘導画像毎にposition embeddingに入力する値を同じにする
        #position_leads = np.append(0,np.repeat(np.arange(1,self.patch_num+1),self.num_lead))
        #position_leads = torch.from_numpy(position_leads).to(x.device)
        
        position_leads = torch.arange(self.max_position_embedding, dtype=torch.long, device=x.device) # トークン数文作成
        position_leads = position_leads.unsqueeze(0).expand_as(torch.zeros(batch_size,self.max_position_embedding)) # batchsize分作られる
        position_embeddings = self.position_embeddings(position_leads)

        # 2つの埋め込みテンソルを足し合わせる [batch_size, seq_len, hidden_size]
        embeddings = x + position_embeddings # + token_type_embeddings

        # LayerNormalizationとDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# Embedding(誘導画像から特徴抽出)にCNNを用いる
class CNN_Embeddings(nn.Module):
    """文章の単語ID列と、1文目か2文目かの情報を、埋め込みベクトルに変換する
    """

    def __init__(self, config):
        super(CNN_Embeddings, self).__init__()

        # 2つのベクトル表現の埋め込み
        self.max_position_embeddings = config.patch_num * config.num_lead + 1 
        
        if config.overlap:
            self.max_position_embedding = 2 * config.num_lead*config.patch_num_v*config.patch_num_h+config.num_lead+1
        else:
            self.max_position_embedding = config.num_lead*config.patch_num_v*config.patch_num_h+config.num_lead+1
            
        self.hidden_size = config.hidden_size
        self.num_att_head = config.num_attention_heads
        
        # 1. 誘導画像をパッチに分割
        # 2. パッチごとにTransformer Encoderに入力する
        # 3. 得られた特徴量を平坦化(or nnにより抽出)し, それを各誘導から得られた特徴量とする (Emdbedding)

        # Transformer Positional Embedding：位置情報テンソルをベクトルに変換
        # Transformerの場合はsin、cosからなる固定値だったが、BERTは学習させる
        self.lead_embeddings = ResNet18(config.local_conf)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 作成したLayerNormalization層
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_leads, attention_show_flg=False):
        '''
        input_leads： [batch_size, number of lead, C, H, W]の文章の各誘導画像のテンソル
        '''

        # 1. Token Embeddings
        # 各誘導から分散表現を獲得
        data = input_leads.permute(1,0,2,3,4) # (batch_size, number of lead, C, H, W) ⇒(number of lead, batch_size, C, H, W) 
        batch_size = data.size(1)
        lead_embeddings = torch.zeros(self.max_position_embeddings,batch_size,self.hidden_size).to(device=input_leads.device) # 最初から全て0でベクトルを作成すれば, 誘導画像がないベクトルは0(pad)として扱える
        
        # 各誘導に対して, 分散表現を獲得する
        # 0は cls tokenのためfor文内で飛ばす. すでに要素0のベクトルとなっている。
        for i in range(1,self.max_position_embeddings): 
            lead_embeddings[i] = self.lead_embeddings(data[i-1]) # data[i-1] : エンコードするのはindex=0からスタート
        
        # (number of lead, batch_size, embed) ⇒ (batch_size, number of lead, embed) 
        lead_embeddings = lead_embeddings.permute(1,0,2)

        # 2. Transformer Positional Embedding：
        # [0, 1, 2 ... 誘導数 + 1(cls)]と誘導数+cls分の数字が1つずつ昇順に入った
        # [batch_size, self.max_position_embeddings] の テンソルposition_leadsを作成
        # position_leadsを入力して、position_embeddings層から分散表現の次元数のテンソルを取り出す
        position_leads = torch.arange(self.max_position_embeddings, dtype=torch.long, device=input_leads.device) # (0,1,2, ... 誘導数+1)
        position_leads = position_leads.unsqueeze(0).expand_as(torch.zeros(batch_size,self.max_position_embeddings)) # (0,1,2,.. 誘導数+1) がbatchsize分作られる
        position_embeddings = self.position_embeddings(position_leads)

        # 2つの埋め込みテンソルを足し合わせる [batch_size, seq_len, hidden_size]
        embeddings = lead_embeddings + position_embeddings # + token_type_embeddings

        # LayerNormalizationとDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class ResNet18(torch.nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        resnet = CNNmodels.resnet18(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLP_Classification(inchanels=resnet.fc.in_features, classification_class=config.hidden_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)


class Layer(nn.Module):
    '''Transformer Encoder Layerモジュールです。Transformerになります'''

    def __init__(self, config):
        super(Layer, self).__init__()

        # Self-Attention部分
        self.attention = Attention(config)

        # Self-Attentionの出力を処理する全結合層
        self.intermediate = Intermediate(config)

        # Self-Attentionによる特徴量とLayerへの元の入力を足し算する層
        self.output = Output(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embedderモジュールの出力テンソル[batch_size, seq_len, hidden_size]
        attention_mask：Transformerのマスクと同じ働きのマスキング
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            attention_output, attention_probs = self.attention(hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]


class Attention(nn.Module):
    '''LayerモジュールのSelf-Attention部分です'''

    def __init__(self, config):
        super(Attention, self).__init__()
        self.selfattn = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor：Embeddingsモジュールもしくは前段のLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


class SelfAttention(nn.Module):
    '''AttentionのSelf-Attentionです'''

    def __init__(self, config):
        super(SelfAttention, self).__init__()
        
        # fusion style
        self.fusion = config.fusion

        self.num_lead = config.num_lead
        self.patch_num_v = config.patch_num_v
        self.patch_num_h = config.patch_num_h
        self.imagesize = config.input_size

        self.num_attention_heads = config.num_attention_heads # Multi head self attention の数 
        # num_attention_heads': 8
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 32/4=8(一つのheadに対応する次元数)
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # = 'hidden_size': 32

        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        if config.overlap:
            patch_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h))
            stride_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h/2))
            self.patches_num = int((self.imagesize[1] - (patch_size[1]-1) -1 )/stride_size[1]+1) # 一つの画像がもつトークン数
            self.max_position_embeddings = self.num_lead * self.patches_num + self.num_lead+1
        else:
            self.patches_num = self.patch_num_v*self.patch_num_h
            self.max_position_embeddings = self.num_lead* self.patches_num + self.num_lead+1 # 誘導画像数*パッチ数 + cls_lead + cls_dx


    def transpose_for_scores(self, x):
        '''multi-head Attention用にテンソルの形を変換する
        [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def make_mask(self,x): # (_)
        # cls_leadの他の誘導に対するAttention scoreを-infにする. forward内softmaxにより-infがAttention scoreを最小にする
        # CLS_dxにおけるマスク. cls_leadに対してmask
        batch_size = x.size(0)

        cls_lead_mask = torch.zeros(batch_size,self.max_position_embeddings).unsqueeze(1)
        # 0: 提案手法, cls_dx cls_lead not attention / 1: cls_dx cls_lead attention not figure patch
        if self.fusion == 0:
            cls_lead_mask[:,:,1:self.num_lead+1] = 1 # cls_dxとcls_leadsがAttentionを計算しない(正しくは, Maskingすることでscoreを0にする), cls_dxは画像patchとAttentionをとる
        elif self.fusion == 1:
            cls_lead_mask[:,:,self.num_lead+1:] = 1 # cls_dxとcls_leadsがAttentionを計算, cls_dxは画像patchとAttentionをとらない
        
        # clsの数(cls_dx(1), cls_lead(19)), 各cls_leadに対するマスクを作成
        # 1の部分がマスクになる
        for i in range(self.num_lead):
            lead_mask = torch.ones(batch_size,self.max_position_embeddings)
            lead_mask[:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 # 各cls_leadにおいて, 関係ないパッチに対してはマスクをかける
            lead_mask = lead_mask.unsqueeze(1)
            cls_lead_mask = torch.cat((cls_lead_mask, lead_mask),dim=1)
        cls_lead_mask = cls_lead_mask.unsqueeze(1)
        cls_lead_mask = cls_lead_mask * -99999.0 # マスク 
        return cls_lead_mask

    def make_mask_2(self,x): #(_2)
        batch_size =x.size(0)
        cls_lead_mask = torch.zeros(batch_size,self.max_position_embeddings).unsqueeze(1)
        cls_lead_mask[:,:,1:self.num_lead+1] = 1 # cls_dx のAttention
        for i in range(self.num_lead): # cls_lead のAttention
            lead_mask = torch.ones(batch_size,self.max_position_embeddings)
            lead_mask[:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 
            lead_mask = lead_mask.unsqueeze(1)
            cls_lead_mask = torch.cat((cls_lead_mask, lead_mask),dim=1)

        for i in range(self.num_lead): # 各パッチトークンのAttention
            lead_mask = torch.ones(batch_size,self.patches_num,self.max_position_embeddings)
            lead_mask[:,:,0] = 0 # cls_dx
            lead_mask[:,:,i+1] = 0 # cls_lead
            lead_mask[:,:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 # パッチトークン 同じ誘導内の画像パッチトークンのみ, 画像パッチトークンが参照
            cls_lead_mask= torch.cat((cls_lead_mask, lead_mask),dim=1)

        cls_lead_mask = cls_lead_mask.unsqueeze(1)
        cls_lead_mask = cls_lead_mask * -99999.0 # マスク 
        return cls_lead_mask

    def make_mask_3(self,x): #(_3)
        batch_size =x.size(0)
        cls_lead_mask = torch.zeros(batch_size,self.max_position_embeddings).unsqueeze(1)
        cls_lead_mask[:,:,1:self.num_lead+1] = 1 # cls_dx のAttention
        for i in range(self.num_lead): # cls_lead のAttention
            lead_mask = torch.ones(batch_size,self.max_position_embeddings)
            lead_mask[:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 
            lead_mask = lead_mask.unsqueeze(1)
            cls_lead_mask = torch.cat((cls_lead_mask, lead_mask),dim=1)

        for i in range(self.num_lead): # 各パッチトークンのAttention
            lead_mask = torch.ones(batch_size,self.patches_num,self.max_position_embeddings)
            lead_mask[:,:,i+1] = 0 # cls_lead
            lead_mask[:,:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 # パッチトークン 同じ誘導内の画像パッチトークンのみ, 画像パッチトークンが参照
            cls_lead_mask= torch.cat((cls_lead_mask, lead_mask),dim=1)

        cls_lead_mask = cls_lead_mask.unsqueeze(1)
        cls_lead_mask = cls_lead_mask * -99999.0 # マスク 
        return cls_lead_mask

    def make_mask_4(self,x): # _4
        batch_size =x.size(0)
        cls_lead_mask = torch.zeros(batch_size,self.max_position_embeddings).unsqueeze(1)
        cls_lead_mask[:,:,1:self.num_lead+1] = 1 # cls_dx のAttention

        for i in range(self.num_lead): # cls_lead のAttention
            lead_mask = torch.ones(batch_size,self.max_position_embeddings)
            lead_mask[:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 
            lead_mask = lead_mask.unsqueeze(1)
            cls_lead_mask = torch.cat((cls_lead_mask, lead_mask),dim=1)

        for i in range(self.num_lead): # 各パッチトークンのAttention
            lead_mask = torch.ones(batch_size,self.patches_num,self.max_position_embeddings)
            lead_mask[:,:,0] = 0 # cls_dx
            lead_mask[:,:,i+1] = 0 # cls_lead
            lead_mask[:,:,self.num_lead+1:] = 0 # パッチトークン 他の誘導を含めた全ての画像パッチトークンを, 画像パッチトークンが参照
            cls_lead_mask = torch.cat((cls_lead_mask, lead_mask),dim=1)

        cls_lead_mask = cls_lead_mask.unsqueeze(1)
        cls_lead_mask = cls_lead_mask * -99999.0 # マスク 
        return cls_lead_mask


    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールもしくは前段のLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        # 入力を全結合層で特徴量変換 multi-head Attention全てをまとめて変換する
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # multi-head Attention用にテンソルの形を変換
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 特徴量同士を掛け算して似ている度合をAttention_scoresとして求める
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # マスクがある部分にマスクをかける
        cls_lead_mask = self.make_mask(hidden_states)
        # masked image modelでのmask : attention_mask,  clsにおけるmask : cls_lead_mask
        attention_scores = attention_scores + attention_mask# + cls_lead_mask.to(hidden_states.device) # make_mask (_2), make_mask_3 (_3)
        attention_scores[:,:,0:self.num_lead+1,:] += cls_lead_mask.to(hidden_states.device) # make_mask_  (_)

        # Attentionを正規化する
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # ドロップアウトします
        attention_probs = self.dropout(attention_probs)

        # Attention Mapを掛け算します
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのテンソルの形をもとに戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # attention_showのときは、attention_probsもリターンする
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class SelfOutput(nn.Module):
    '''SelfAttentionの出力を処理する全結合層です'''

    def __init__(self, config):
        super(SelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 'hidden_dropout_prob': 0.1

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states：SelfAttentionの出力テンソル
        input_tensor：Embeddingsモジュールもしくは前段のLayerからの出力
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    '''Gaussian Error Linear Unitという活性化関数です。
    LeLUが0でカクっと不連続なので、そこを連続になるように滑らかにした形のLeLUです。
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Intermediate(nn.Module):
    '''のTransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(Intermediate, self).__init__()

        # 全結合層：'hidden_size': 256, 'intermediate_size': 1024
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # 活性化関数gelu
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states： Attentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # GELUによる活性化
        return hidden_states


class Output(nn.Module):
    '''TransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(Output, self).__init__()

        # 全結合層：'intermediate_size': 1024, 'hidden_size': 256
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

        # 'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states： Intermediateの出力テンソル
        input_tensor：Attentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Layerモジュールの繰り返し部分モジュールの繰り返し部分です


class Encoder(nn.Module):
    def __init__(self, config):
        '''Layerモジュールの繰り返し部分モジュールの繰り返し部分です'''
        super(Encoder, self).__init__()

        # config.num_hidden_layers 個のLayerモジュールを作ります
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：返り値を全TransformerBlockモジュールの出力にするか、
        それとも、最終層だけにするかのフラグ。
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # 返り値として使うリスト
        all_encoder_layers = []
        # 各ブロックのattention probsのリスト
        all_attention_probs = []
        # Layerモジュールの処理を繰り返す
        for layer_module in self.layer:

            if attention_show_flg == True:
                '''attention_showのときは、attention_probsもリターンする'''
                hidden_states, attention_probs = layer_module(hidden_states, attention_mask, attention_show_flg)
                all_attention_probs.append(attention_probs)
            elif attention_show_flg == False:
                hidden_states = layer_module(hidden_states, attention_mask, attention_show_flg)

            # 返り値にLayerから出力された特徴量を8層分、すべて使用する場合の処理
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 返り値に最後のLayerから出力された特徴量だけを使う場合の処理
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            all_attention_probs = torch.stack(all_attention_probs).permute(1,0,2,3,4) # (batch, block, num head, num token, num token)

        # attention_showのときは、attention_probsもリターンする
        if attention_show_flg == True:
            return all_encoder_layers, all_attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers


class Pooler(nn.Module):
    '''入力心電図のclsの特徴量を変換して保持するためのモジュール'''

    def __init__(self, config):
        super(Pooler, self).__init__()

        # 全結合層、'hidden_size': 256
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # clsの特徴量を取得
        first_token_tensor = hidden_states[:, 0]

        # 全結合層で特徴量変換
        pooled_output = self.dense(first_token_tensor)

        # 活性化関数Tanhを計算
        pooled_output = self.activation(pooled_output)

        return pooled_output


class TE(nn.Module):
    '''モジュールを全部つなげたモデル'''

    def __init__(self, config):
        super(TE, self).__init__()    
        # 3つのモジュールを作成
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.pooler = Pooler(config)
        
        self.imagesize = config.input_size
        self.num_lead = config.num_lead
        self.patch_num_v = config.patch_num_v
        self.patch_num_h = config.patch_num_h
        
        if config.overlap:
            patch_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h))
            stride_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h/2))
            self.max_position_embeddings = self.num_lead * int((self.imagesize[1] - (patch_size[1]-1) -1 )/stride_size[1]+1) + self.num_lead+1
        else:
            self.max_position_embeddings = self.num_lead*self.patch_num_v*self.patch_num_h+(self.num_lead+1) # 誘導画像数*パッチ数 + cls_lead + cls_dx

    def forward(self, input_leads, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_leads： [batch_size, number of lead, C, H, W]の文章の各誘導画像のテンソル
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力にTransformerの全段の出力をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        if attention_mask is None:
            attention_mask = torch.ones_like(torch.zeros(input_leads.size(0),self.max_position_embeddings)) 

        # マスクの変形　[minibatch, 1, 1, seq_length]にする
        # 後ほどmulti-head Attentionで使用できる形にしたいので
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # マスクは0、1だがソフトマックスを計算したときにマスクになるように、0と-infにする
        # -infの代わりに-10000にしておく
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32,device=input_leads.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 順伝搬させる
        # Embeddinsモジュール
        embedding_output = self.embeddings(input_leads,attention_show_flg=attention_show_flg)

        # Layerモジュール（Transformer）を繰り返すEncoderモジュール
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''

            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # Poolerモジュール
        # encoderの一番最後のLayerから出力された特徴量を使う
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layersがFalseの場合はリストではなく、テンソルを返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output,embedding_output

# 学習済みモデルのロード
def set_learned_params(net, weights_path = "./weights/pytorch_model.bin"):

    # セットするパラメータを読み込む
    loaded_state_dict = torch.load(weights_path)

    # 現在のネットワークモデルのパラメータ名
    net.eval()
    param_names = []  # パラメータの名前を格納していく

    for name, param in net.named_parameters():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = net.state_dict().copy()

    # 新たなstate_dictに学習済みの値を代入
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる
        print(str(key_name)+"→"+str(name))  # 何から何に入ったかを表示

        # 現在のネットワークのパラメータを全部ロードしたら終える
        if (index+1 - len(param_names)) >= 0:
            break

    # 新たなstate_dictを構築したモデルに与える
    net.load_state_dict(new_state_dict)

    return net

# --------------------------------------------------------------------------------------------------------
# 事前学習課題：Masked Image Model用のモジュール
class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        '''事前学習課題：Masked Image Model用のモジュール
        '''
        super(MaskedWordPredictions, self).__init__()

        # モデルから出力された特徴量を変換するモジュール（入出力のサイズは同じ）
        self.transform = PredictionHeadTransform(config)

        # self.transformの出力から、各位置の単語がどれかを当てる全結合層
        self.decoder = nn.Linear(in_features=config.hidden_size,  # 'hidden_size': 32
                                 out_features=config.hidden_size,  # 'hidden_size' : 32 もとの特徴量を当てる
                                 bias=False)
        # バイアス項
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))  # 'vocab_size': 32

    def forward(self, hidden_states):
        '''
        hidden_states：モデルからの出力[batch_size, seq_len, hidden_size]
        '''
        # モデルから出力された特徴量を変換
        # 出力サイズ：[batch_size, seq_len, hidden_size]
        hidden_states = self.transform(hidden_states)

        # 各位置の単語がボキャブラリーのどの単語なのかをクラス分類で予測
        # 出力サイズ：[batch_size, seq_len, vocab_size]
        hidden_states = self.decoder(hidden_states) + self.bias

        # mask部分のみとりだしたい

        return hidden_states


class PredictionHeadTransform(nn.Module):
    '''MaskedWordPredictionsにて、modelの出力する特徴量を変換するモジュール（入出力のサイズは同じ）'''

    def __init__(self, config):
        super(PredictionHeadTransform, self).__init__()

        # 全結合層 'hidden_size': 32
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # 活性化関数gelu
        self.transform_act_fn = gelu

        # LayerNormalization
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        '''hidden_statesはsequence_output:[minibatch, seq_len, hidden_size]'''
        # 全結合層で特徴量変換し、活性化関数geluを計算したあと、LayerNormalizationする
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MaskedIM(nn.Module):
    '''モデルに、事前学習課題用のアダプターモジュール   PreTrainingHeadsをつなげたモデル'''

    def __init__(self, config, net):
        super(MaskedIM, self).__init__()

        # モジュール
        self.net = net  # モデル

        # 事前学習課題用のアダプターモジュール
        # 全結合 2層
        self.cls = MaskedWordPredictions(config)

    def forward(self, input_leads, token_type_ids=None, attention_mask=None):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        '''

        # 基本モデル部分の順伝搬
        encoded_layers, pooled_output,emmbeding = self.net(input_leads, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)

        # 事前学習課題の推論を実施
        prediction_scores = self.cls(encoded_layers)

        return prediction_scores, pooled_output, emmbeding


# 学習済みモデルのロード
def set_learned_params(net, weights_path = "./weights/pytorch_model.bin"):

    # セットするパラメータを読み込む
    loaded_state_dict = torch.load(weights_path)

    # 現在のネットワークモデルのパラメータ名
    net.eval()
    param_names = []  # パラメータの名前を格納していく

    for name, param in net.named_parameters():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = net.state_dict().copy()

    # 新たなstate_dictに学習済みの値を代入
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる
        print(str(key_name)+"→"+str(name))  # 何から何に入ったかを表示

        # 現在のネットワークのパラメータを全部ロードしたら終える
        if (index+1 - len(param_names)) >= 0:
            break

    # 新たなstate_dictを構築したモデルに与える
    net.load_state_dict(new_state_dict)

    return net




