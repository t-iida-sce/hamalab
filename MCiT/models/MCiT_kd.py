
# 参考した実装
# https://github.com/huggingface/pytorch-pretrained-BERT
# 細かな点を Vision Transformer に変更
# 必要なパッケージのimport
import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


# LayerNormalization層を定義
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization層 """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# Embeddingsモジュール
class Embeddings(nn.Module):
    """画像からトークンにエンコード """

    def __init__(self, config,channels = 3):
        super(Embeddings, self).__init__()

        # 2つのベクトル表現の埋め込み
        self.hidden_size = config.hidden_size
        self.num_att_head = config.num_attention_heads
        self.imagesize = config.input_size
        self.patch_num_v = config.patch_num_v
        self.patch_num_h = config.patch_num_h
        self.num_lead = config.num_lead
        self.pos_embd= config.pos_embd
        self.cls_tokens = nn.Parameter(torch.zeros(1, 1 + self.num_lead, config.hidden_size)) # cls_dx, cls_lead を作成
        
        # 1. 誘導画像をパッチに分割
        # 2. パッチごとにTransformer Encoderに入力する
        # 3. 得られた特徴量を平坦化(or nnにより抽出)し, それを各誘導から得られた特徴量とする (Emdbedding)

        # 画像パッチに線形変換する
        patch_size = (int(self.imagesize[0]/self.patch_num_v), int(self.imagesize[1]/self.patch_num_h)) # 画像における列方向に分割, 分割された数がembeddingする数となる
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=config.hidden_size,kernel_size=patch_size,stride=patch_size,bias=False)

        if self.pos_embd == 0:  # 全てのチャートに対して, 昇順のパラメータを作成, 位置埋め込み
            self.max_position_num = self.num_lead*self.patch_num_v*self.patch_num_h+self.num_lead+1                                            
            self.position_embeddings = nn.Embedding(self.max_position_num, config.hidden_size)
        else:         # 単一チャートに対して, 昇順のパラメータを作成, 位置埋め込み
            self.patch_position_num = self.patch_num_v*self.patch_num_h
            self.patch_position_embeddings = nn.Embedding(self.patch_position_num, config.hidden_size)
            self.clsseg_position_num= self.num_lead+1 
            self.clsseg_position_embeddings = nn.Embedding(self.clsseg_position_num, config.hidden_size)

        # 作成したLayerNormalization層
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_leads,attention_show_flg=False):
        '''
        input_leads： [batch_size, number of lead, C, H, W]の文章の各誘導画像のテンソル
        attention_show_flag: visualize attention flag
        '''

        # 1. Token Embeddings 
        batch_size = input_leads.size(0)
        input_leads = input_leads.permute(0,2,1,3,4) # (batch_size, number of lead, C, H, W) ⇒ (batch_size, C, number of lead,H, W )
        # Mask input
        
        input_leads = input_leads.reshape(batch_size, input_leads.size(1), -1, input_leads.size(4)) #(batch_size, C, number of lead*H, W)
        
        # 診断用のcls_dx , 誘導用のcls_leadを用意する
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)

        x = self.patch_embedding(input_leads) # パッチを抽出し, tokenに
        x = x.flatten(2) # 2次元に変換
        x = x.transpose(-1, -2) 
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embd == 0:
            position_embds = torch.arange(self.max_position_num, dtype=torch.long, device=x.device) # トークン数分作成
            position_embds = position_embds.unsqueeze(0).expand_as(torch.zeros(batch_size,self.max_position_num)) # batchsize分作られる
            position_embeddings = self.position_embeddings(position_embds)
        else: 
            p_pos_embds = torch.arange(self.patch_position_num, dtype=torch.long, device=x.device) # 単一viewｍのトークン数分作成
            p_pos_embds = p_pos_embds.unsqueeze(0).expand_as(torch.zeros(batch_size,self.patch_position_num)) # batchsize分作られる
            p_position_embeddings = self.patch_position_embeddings(p_pos_embds)
            p_position_embeddings = p_position_embeddings.repeat(1,self.num_lead,1) # 入力トークン数に拡張
            
            cs_pos_embds =  torch.arange(self.clsseg_position_num, dtype=torch.long, device=x.device) # 単一viewｍのトークン数分作成
            cs_pos_embds = cs_pos_embds.unsqueeze(0).expand_as(torch.zeros(batch_size,self.clsseg_position_num)) # batchsize分作られる
            cs_position_embeddings = self.clsseg_position_embeddings(cs_pos_embds)
            
            position_embeddings = torch.cat((cs_position_embeddings,p_position_embeddings),dim=1)

        # 2つの埋め込みテンソルを足し合わせる [batch_size, seq_len, hidden_size]
        embeddings = x + position_embeddings # + token_type_embeddings
             
        # LayerNormalizationとDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_embeddings


class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()

        # Self-Attention部分
        self.attention = Attention(config)

        # Self-Attentionの出力を処理する全結合層
        self.intermediate = Intermediate(config)


    def forward(self, hidden_states, attention_show_flg=False):
        '''
        hidden_states：Embedderモジュールの出力テンソル[batch_size, seq_len, hidden_size]
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            attention_output, attention_probs = self.attention(hidden_states, attention_show_flg)
            hidden_states = hidden_states + attention_output
            intermediate_output = self.intermediate(hidden_states)
            layer_output = hidden_states + intermediate_output
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(hidden_states, attention_show_flg)
            hidden_states = hidden_states + attention_output
            intermediate_output = self.intermediate(hidden_states)
            layer_output = hidden_states + intermediate_output
            return layer_output  # [batch_size, seq_length, hidden_size]


class Attention(nn.Module):
    '''LayerモジュールのSelf-Attention部分です'''

    def __init__(self, config):
        super(Attention, self).__init__()
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.selfattn = SelfAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_show_flg=False):
        '''
        input_tensor：Embeddingsモジュールもしくは前段のLayerからの出力
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            hidden_states = self.LayerNorm(input_tensor)
            attention_output, attention_probs = self.selfattn(hidden_states,attention_show_flg)
            attention_output = self.dropout(attention_output)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            hidden_states = self.LayerNorm(input_tensor)
            attention_output = self.selfattn(hidden_states, attention_show_flg)
            attention_output = self.dropout(attention_output)
            return attention_output


class SelfAttention(nn.Module):
    '''AttentionのSelf-Attentionです'''

    def __init__(self, config):
        super(SelfAttention, self).__init__()
        
        # attention style
        self.fusion = config.fusion
        self.howatt = config.how_att

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

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
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
        # 0: 提案手法, cls SEG not attention / 1: cls SEG attention not figure patch
        if self.fusion == 0:
            cls_lead_mask[:,:,1:self.num_lead+1] = 1 # cls_dxとcls_leadsがAttentionを計算しない(正しくは, Maskingすることでscoreを0にする), cls_dxは画像patchとAttentionをとる
        elif self.fusion == 1:
            cls_lead_mask[:,:,self.num_lead+1:] = 1 # cls_dxとcls_leadsがAttentionを計算, cls_dxは画像patchとAttentionをとらない
        
        # clsの数(cls_dx(1), cls_lead(19)), 各cls_leadに対するマスクを作成
        # 1の部分がマスクになる
        for i in range(self.num_lead):
            lead_mask = torch.ones(batch_size,self.max_position_embeddings)
            lead_mask[:,i+1] = 0
            lead_mask[:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 # 各cls_leadにおいて, 関係ないパッチに対してはマスクをかける
            lead_mask = lead_mask.unsqueeze(1)
            cls_lead_mask = torch.cat((cls_lead_mask, lead_mask),dim=1)

        # 画像パッチトークンに対するAttentionについて
        if self.howatt == 0:
            pass
        elif self.howatt == 1:
            for i in range(self.num_lead): # 各パッチトークンのAttention
                lead_mask = torch.ones(batch_size,self.patches_num,self.max_position_embeddings)
                lead_mask[:,:,0] = 0 # cls_dxにattention
                lead_mask[:,:,i+1] = 0 # cls_leadにattention
                lead_mask[:,:,(self.num_lead+1)+self.patches_num*i:(self.num_lead+1)+self.patches_num*(i+1)] = 0 # パッチトークン 同じ誘導内の画像パッチトークンのみ, 画像パッチトークンが参照
                cls_lead_mask= torch.cat((cls_lead_mask, lead_mask),dim=1)

        cls_lead_mask = cls_lead_mask.unsqueeze(1)
        cls_lead_mask = cls_lead_mask * -99999.0 # マスク 
        return cls_lead_mask


    def forward(self, hidden_states, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールもしくは前段のLayerからの出力
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
        # masked image modelingにおけるmask : mim
        # attentionの制限
        if self.howatt == 0:
            attention_scores[:,:,0:self.num_lead+1,:] += cls_lead_mask.to(hidden_states.device) # how_att == 1
        else:
            attention_scores  += cls_lead_mask.to(hidden_states.device) # # how_att == 2,3,4

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

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

def gelu(x):
    '''Gaussian Error Linear Unitという活性化関数です。
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Intermediate(nn.Module):
    '''のTransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(Intermediate, self).__init__()

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states： Attentionの出力テンソル
        '''
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # GELUによる活性化
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# Layerモジュールの繰り返し部分モジュールの繰り返し部分です

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールの出力
        output_all_encoded_layers：返り値を全TransformerBlockモジュールの出力にするか、
        それとも、最終層だけにするかのフラグ
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
                hidden_states, attention_probs = layer_module(hidden_states, attention_show_flg)
                all_attention_probs.append(attention_probs)
            elif attention_show_flg == False:
                hidden_states = layer_module(hidden_states, attention_show_flg)

            # 返り値にLayerから出力された特徴量を層分、すべて使用する場合の処理
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

class te(nn.Module):
    '''モジュールを全部つなげたモデル'''

    def __init__(self, config):
        super(te, self).__init__()    
        # 3つのモジュールを作成
        self.encoder = Encoder(config)
        
        self.imagesize = config.input_size
        self.num_lead = config.num_lead
        self.patch_num_v = config.patch_num_v
        self.patch_num_h = config.patch_num_h
        
        self.max_position_embeddings = self.num_lead*self.patch_num_v*self.patch_num_h+(self.num_lead+1) # 誘導画像数*パッチ数 + cls_lead + cls_dx

    def forward(self, input_leads, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_leads： [batch_size, number of lead, C, H, W]の文章の各誘導画像のテンソル
        output_all_encoded_layers：最終出力にTransformerの全段の出力をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # Layerモジュール（Transformer）を繰り返すEncoderモジュール
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            encoded_layers, attention_probs = self.encoder(input_leads,output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(input_leads,output_all_encoded_layers, attention_show_flg)

        # output_all_encoded_layersがFalseの場合はリストではなく、テンソルを返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return encoded_layers, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, input_leads

class ClsAttention(nn.Module):
    def __init__(self, config):
        super(ClsAttention, self).__init__()

        self.num_lead = config.num_lead
        self.patch_num_v = config.patch_num_v
        self.patch_num_h = config.patch_num_h
        self.imagesize = config.input_size
        self.cls_hidden_size = config.cls_hidden_size

        self.num_attention_heads = config.num_attention_heads # Multi head self attention の数 
        # num_attention_heads': 8
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 32/4=8(一つのheadに対応する次元数)
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # = 'hidden_size': 32

        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.affine = nn.Linear(config.hidden_size, self.cls_hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.max_position_embeddings = self.num_lead+1 # 画像数 + cls

    def transpose_for_scores(self, x):
        '''multi-head Attention用にテンソルの形を変換する
        [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールもしくは前段のLayerからの出力
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
        
        att_vectors = attention_scores[:,:,0,1:self.num_lead+1].mean(1).unsqueeze(2)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのテンソルの形をもとに戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        context_layer = self.affine(context_layer[:,1:self.num_lead+1,:])
        cls_vector = torch.cat((att_vectors,context_layer),dim=2).view(hidden_states.size(0),-1)
        
        return cls_vector

        
class MLPHead(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=False):
        super(MLPHead, self).__init__()
        if not mid_channels:
            mid_channels = in_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)
        
class MCiT(nn.Module):

    def __init__(self, config,task=5):
        super(MCiT, self).__init__()

        self.config = config

        # モジュール
        self.embeddings = Embeddings(config.model_conf) # 埋め込み
        self.te = te(config.model_conf)  # Transformer Encoderモデル
        self.cls_feature = config.classifier_conf.cls
        self.num_lead = config.model_conf.num_lead
        
        self.g_layer = config.kd_conf.g_layer
        self.kd_linear = config.kd_conf.linear
        if config.kd_conf.linear==0:
            pass
        elif config.kd_conf.linear==1:
            self.kd_layer = nn.Linear(in_features=config.model_conf.hidden_size, out_features=config.model_conf.hidden_size)
        elif config.kd_conf.linear==2:
            self.kd_layer = MLPHead(in_channels=config.model_conf.hidden_size,mid_channels=config.model_conf.hidden_size, out_channels=config.model_conf.hidden_size)

        # 予測器　
        if config.classifier_conf.cls == 0:
            self.features_ = config.model_conf.hidden_size
            self.clssifier_cls = nn.Linear(in_features=self.features_, out_features=config.classifier_conf.feature)
        elif config.classifier_conf.cls == 1:
            self.features_ = config.model_conf.hidden_size*(1 + self.num_lead)
            self.clssifier_cls = nn.Linear(in_features=self.features_, out_features=config.classifier_conf.feature)
        elif config.classifier_conf.cls == 2:
            self.features_ = config.model_conf.hidden_size*(1 + self.num_lead)
            self.clssifier_cls = MLPHead(in_channels=self.features_,mid_channels=int(self.features_/3), out_channels=config.classifier_conf.feature)
        elif config.classifier_conf.cls == 3:
            self.features_ = config.model_conf.hidden_size*2
            self.clssifier_cls = nn.Linear(in_features=self.features_, out_features=config.classifier_conf.feature)
        elif config.classifier_conf.cls == 4:
            self.features_= self.num_lead + self.num_lead * config.model_conf.cls_hidden_size
            self.clssifier_cls = nn.Sequential(
                ClsAttention(config.model_conf),
                nn.Flatten(),
                nn.Linear(in_features=self.features_, out_features=config.classifier_conf.feature)
            )
        elif config.classifier_conf.cls == 5:
            self.features_ = config.model_conf.hidden_size*2
            self.clssifier_cls =MLPHead(in_channels=self.features_,mid_channels=self.features_, out_channels=config.classifier_conf.feature)

    def forward(self, input_leads, output_all_encoded_layers=True, attention_show_flg=False, output_all_encoded_layers_show=False):
        '''
        input_leads： [batch_size, sequence_length]の各誘導の羅列
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # Embedding
        embedding_output,_ = self.embeddings(input_leads,attention_show_flg=attention_show_flg)

        # teの基本モデル部分の順伝搬
        # 順伝搬させる
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            encoded_layers, attention_probs = self.te(
                embedding_output, output_all_encoded_layers=output_all_encoded_layers, attention_show_flg=attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, embed = self.te(
                embedding_output, output_all_encoded_layers=output_all_encoded_layers, attention_show_flg=attention_show_flg)
            
        # Encoderから出力される特徴量を使用して分類
        encoded_layers = torch.stack(encoded_layers) # [layers, batchsize, tokens, tokens vec]
        if self.cls_feature == 0:
            vec_cls = encoded_layers[-1,:,:, 0, :].view(-1, self.features_)
            out = self.clssifier_cls(vec_cls)
        elif self.cls_feature == 1:
            vec_cls = encoded_layers[-1,:, 0:self.num_lead+1, :].view(-1, self.features_)
            out = self.clssifier_cls(vec_cls)
        elif self.cls_feature == 2:
            vec_cls = encoded_layers[-1,:, 0:self.num_lead+1, :].view(-1, self.features_)
            out = self.clssifier_cls(vec_cls)
        elif self.cls_feature == 3:
            pool_view = encoded_layers[-1,:, 1:self.num_lead+1, :]
            pool_cls = torch.mean(pool_view,1)
            vec_cls = torch.cat((encoded_layers[-1,:, 0, :],pool_cls),dim=1)
            out = self.clssifier_cls(vec_cls)
        elif self.cls_feature == 4:
            out = self.clssifier_cls(encoded_layers)
        elif self.cls_feature == 5:
            pool_view = encoded_layers[-1,:, 1:self.num_lead+1, :]
            pool_cls = torch.mean(pool_view,1)
            vec_cls = torch.cat((encoded_layers[-1,:, 0, :],pool_cls),dim=1)
            out = self.clssifier_cls(vec_cls)
        
        # guided tokenに対する処理
        if self.kd_linear:
            for i in range(1,self.config.model_conf.num_lead+1): 
                encoded_layers[self.g_layer,:,i,:] = self.kd_layer(encoded_layers[self.g_layer,:,i,:].clone())

        if output_all_encoded_layers_show:
            return encoded_layers
        
        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out,encoded_layers[self.g_layer,:,:,:]







