import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score, precision_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix 

from models import TransformerEncoder

class TEForPred(nn.Module):

    def __init__(self, net,config,task=5):
        super(TEForPred, self).__init__()

        # MLiTモジュール
        self.te = net  # Transformer Encoderモデル
        self.config = config
        self.cls_feature = config.classifier_conf.cls

        # 線形予測器
        if config.classifier_conf.cls == 0:
            self.features_ = config.model_conf.hidden_size
            self.clssifier_cls = nn.Linear(in_features=self.features_, out_features=task)
        elif config.classifier_conf.cls == 1:
            self.features_ = config.model_conf.hidden_size*20
            self.clssifier_cls = nn.Linear(in_features=self.features_, out_features=task)
        else:
            self.clssifier_cls = nn.Linear(in_features=config.model_conf.hidden_size, out_features=task)
            self.clssifier_leads = nn.ModuleList([nn.Linear(in_features=config.model_conf.hidden_size, out_features=task) for _ in range(config.num_lead)])

        # 重み初期化処理
        nn.init.normal_(self.clssifier_cls.weight, std=0.02)
        nn.init.normal_(self.clssifier_cls.bias, 0)

    def forward(self, input_leads, attention_mask=None, output_all_encoded_layers=False, attention_show_flg=False):
        '''
        input_leads： [batch_size, sequence_length]の各誘導の羅列
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # teの基本モデル部分の順伝搬
        # 順伝搬させる
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            encoded_layers, pooled_output, attention_probs = self.te(
                input_leads, attention_mask=attention_mask, output_all_encoded_layers=output_all_encoded_layers, attention_show_flg=attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, pooled_output, embed = self.te(
                input_leads, attention_mask=attention_mask, output_all_encoded_layers=output_all_encoded_layers, attention_show_flg=attention_show_flg)

        # 入力の[CLS]の特徴量を使用して分類
        if self.cls_feature == 0:
            vec_cls = encoded_layers[:, 0, :]
        elif self.cls_feature == 1:
            vec_cls = encoded_layers[:, 0:20, :]

        # cls識別
        vec_cls = vec_cls.view(-1, self.features_)  # sizeを[batch_size, hidden_size]に変換
        out = self.clssifier_cls(vec_cls)

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out,embed

# trainer
def train_model(model, train_loader, test_loader,num_epochs,optimizer,criterion,save_dir,early_stop=100):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    model.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    #torch.backends.cudnn.benchmark = True

    history = []
    test_loss_min = np.Inf

    # epochのループ
    for epoch in range(num_epochs):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0

        model.eval()
        for i,(ecg,l,p,num) in enumerate(train_loader):

            ecg = ecg.to(device)
            l = l.to(device)
            # teForPredに入力
            outputs, _ = model(ecg, attention_mask=None)

            loss = criterion(outputs, l)  # 損失を計算
            train_loss += loss.detach().cpu().numpy().item() * outputs.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy by finding max log probability
            _, pred = torch.max(outputs,dim=1) 
            train_acc += torch.sum(pred == l.data)

        # epochごとのlossと正解率
        train_acc = train_acc.double() / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            for ecg,l,p,num in test_loader:
                ecg = ecg.to(device)
                l = l.to(device)
                # teForPredに入力
                outputs,_ = model(ecg, attention_mask=None)
                loss = criterion(outputs, l)  # 損失を計算
                test_loss += loss.detach().cpu().numpy().item() * outputs.size(0)

                _, pred = torch.max(outputs,dim=1) 
                test_acc += torch.sum(pred == l.data)
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_acc / len(test_loader.dataset)

        if test_loss < test_loss_min:
            test_loss_min = test_loss
            best_epoch = epoch
            best_acc = test_acc
            early_model = copy.deepcopy(model)
            torch.save(early_model.state_dict(), os.path.join(save_dir, 'mlit_besttestloss.pth'))
            print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {test_loss_min:.2f} and acc: {100 * best_acc:.2f}%')
                
        print('Epoch {}/{} | train/test Loss: {:.4f}/{:.4f} Acc: {:.4f}/{:.4f}'.format(epoch+1, num_epochs,train_loss,test_loss,train_acc,test_acc))

        history.append([train_loss,test_loss,train_acc.detach().cpu().item(), test_acc.detach().cpu().item()])

    history = np.array(history)
    save_history(history,save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'mlit.pth'))
    return model, history

def save_history(history,save_dir):
    plt.rcParams.update(plt.rcParamsDefault)

    his_loss = history[:,[0,1]]
    plt.plot(his_loss[:,0],label='train_loss')
    plt.plot(his_loss[:,1],label='test_loss')
    plt.ylim(0,2.0)
    plt.legend()
    plt.title('loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('{}/loss.png'.format(save_dir))
    plt.close()

    his_acc = history[:,[2,3]]
    plt.plot(his_acc[:,0],label='train_acc')
    plt.plot(his_acc[:,1],label='test_acc')
    plt.ylim(0.30,1)
    plt.legend(loc='lower right')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('{}/accuracy.png'.format(save_dir))
    plt.close()

# TE用
def classifier_accuracy(config,te_model,net_trained,train_dataloader,test_dataloader,save_dir,label_name,task):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 訓練終了時の正解率(acc)とF1-scoreを計算
    train_acc,train_f1_score, train_report, train_bi_acc,train_bi_rec,train_bi_pre = accuracy(device,net_trained,train_dataloader,'train',save_dir, label_name,config)
    test_acc,test_f1_score, test_report, test_bi_acc,test_bi_rec, test_bi_pre = accuracy(device,net_trained,test_dataloader,'test',save_dir,label_name,config)
    print('train : acc {:.3f} f1_score {:.3f}'.format(train_acc,train_f1_score))
    print('test : acc {:.3f} f1_score {:.3f}'.format(test_acc,test_f1_score))

    # テストに対する最小loss時の正解率(acc)とF1-scoreを計算
    net_early = TEForPred(te_model,config,task).to(device)
    net_early.load_state_dict(torch.load(os.path.join(save_dir, 'mlit_besttestloss.pth')))
    e_train_acc,e_train_f1_score, e_train_report,e_train_bi_acc,e_train_bi_rec, e_train_bi_pre = accuracy(device,net_early,train_dataloader,'early_train',save_dir,label_name,config)
    e_test_acc,e_test_f1_score, e_test_report,e_test_bi_acc,e_test_bi_rec,e_test_bi_pre = accuracy(device,net_early,test_dataloader,'early_test',save_dir,label_name,config)
    print('early_train : acc {:.3f} f1_score {:.3f}'.format(e_train_acc,e_train_f1_score))
    print('early_test : acc {:.3f} f1_score {:.3f}'.format(e_test_acc,e_test_f1_score))

    # 部位識別に対する訓練acc,訓練f値, テストacc,テストf値の4つと, ACS識別(2class)に対する訓練acc,訓練f値, テストacc,テストf値の4つをまとめて返す
    return e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score, e_train_bi_acc,e_train_bi_rec,e_train_bi_pre,e_test_bi_acc,e_test_bi_rec,e_test_bi_pre

def accuracy(device,model_,loader,eval,save_dir,label_name,config,attention_show_flg=False):
    preds = np.array([]) #予測ラベル
    targets = np.array([]) #正解ラベル
    patients = np.array([]) #患者番号
    result_history = np.array([]) #誤診断した患者番号
    attention=np.array([])
    attention_leads = np.array([]) # 各誘導におけるAttention可視化
    attention_leads_index = np.array([]) # Attentionが大きかった誘導indexを保存
    num_leads = config.model_conf.num_lead * config.model_conf.patch_num_v * config.model_conf.patch_num_h + 1
    with torch.no_grad():
        model_.eval()
        #corrects=0
        for ecg, label,p,num in loader:
            ecg = ecg.to(device)
            l = label.to(device)
            
            outputs, attention_map = model_(ecg, attention_mask=None,attention_show_flg=True)
            if attention_show_flg == True:
                g_att = multi_norm_attention(attention_map,config) # globalにおけるattention map
                # att.argsort()[::] : 誘導に対するattの値が小さい順でインデックスとなるarrayを獲得 ⇒ [0,0,-5:]で最初の患者, CLSにおけるattの降順の誘導indexを得ることができる
                attention_leads_index=np.append(attention_leads_index,g_att.argsort()[:,:,::-1])

            _, pred = torch.max(outputs, 1)  # ラベルを予測

            preds = np.append(preds, pred.cpu().data.numpy())
            targets = np.append(targets,label.cpu().data.numpy())
            patients = np.append(patients,p)
        
        for i in range(len(targets)):
            result_history=np.append(result_history,[patients[i],'0',targets[i],preds[i]])
            if targets[i] != preds[i]:
                result_history=np.append(result_history,[patients[i],'1',targets[i],preds[i]])
        result_history =  result_history.reshape(-1,4)
        pd.DataFrame(result_history,columns=['patients','result','target','prediction']).to_csv('{}/{}_miss_patients.csv'.format(save_dir,eval))
        test_acc = accuracy_score(targets,preds)
        f1_score_ = f1_score(targets, preds,average='macro')
        report = classification_report(targets,preds,output_dict=True)
        
        cm = confusion_matrix(targets, preds)
        create_confusion_matrix(cm,label_name,test_acc,f1_score_,report,eval,save_dir)

        # 2値分類用
        preds_2 = np.where(preds==3, 0, 1)
        targets_2 = np.where(targets==3, 0, 1)
        test_acc2 = accuracy_score(targets_2,preds_2)
        recall2 = recall_score(targets_2,preds_2)
        precision2 = precision_score(targets_2,preds_2)
        report_2 = classification_report(targets_2,preds_2,output_dict=True)
        
        cm_2 = confusion_matrix(targets_2, preds_2)
        create_confusion_matrix_bi(cm_2,['Normal','ACS'],test_acc2,recall2,precision2,report_2,eval+'_2class',save_dir)
        
        return test_acc, f1_score_, report, test_acc2, recall2, precision2

# multi head attentionの重みを全て加算し, 正規化する関数
def multi_norm_attention(normlized_weights,config):
    att = np.array([])
    # Attentionの重みを抽出と規格化
    # Multi-Head Attentionを取り出す
    for index in range(normlized_weights.size(0)): # index = batch_size
        attens=torch.zeros(normlized_weights.size(2),normlized_weights.size(3)) # (leadの数 , 特定のleadのあるleadに対する重み)
        for i in range(config.model_conf.num_attention_heads): # multi headのindex
            attens += normlized_weights[index, i, :, :].detach().cpu() # 特定のbatchにおけるmulti head の重みを(20,20)で得て加算していく
        
        attens /= attens.max() # 加算し終わったものを最大値で割る(正規化)
        att = np.append(att,attens.detach().cpu()) # 得られたattentionの重みをリストに入れる
    att = att.reshape(normlized_weights.size(0),normlized_weights.size(2),normlized_weights.size(3))
    return att

def visualization_lead_attention(ecg,att_weights,g_att,pred,label,p,save_dir,eval):
    att = np.array([])
    # 各誘導のattentionにおけるnulti-headを加算し正規化
    for lead_ in range(att_weights.size(1)):
        att = np.append(att,multi_norm_attention(att_weights[:,lead_,:,:]))
    att = att.reshape(att_weights.size(1),att_weights.size(0),att_weights.size(3),att_weights.size(4))
    att = att.transpose(1,0,2,3)
    visualize_ecg(ecg,att,g_att,pred,label,p,save_dir,eval)
    return 

# ECGにattentionを重ねた画像を出力する
def visualize_ecg(ecg,att,g_att,pred,label,p,save_dir,eval):
    p_name = ['前壁','下壁','側壁','正常','後壁']
    l_name = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','V3R','V4R','V5R','V6R','V7','V8','V9']
    if not os.path.exists(os.path.join(save_dir,eval)):
        os.mkdir(os.path.join(save_dir,eval))
    att = torch.from_numpy(att)
    for i in range(ecg.size(0)): # batch
        fig,ax = plt.subplots(6,4,figsize = (16, 24),sharey = "all", tight_layout=True)
        g_att_index = np.sum(g_att[i,:,:],1)
        g_att_index = g_att_index.argsort()[::-1]
        for l in range(ecg.size(1)): # 誘導
            ecg_ = transforms.functional.to_pil_image(ecg[i,l,:,:,:])

            # mapの作成
            attention_map = torch.sum(att[i,l,:,:],1)
            attmap = (attention_map-attention_map.min())/(attention_map.max()-attention_map.min()) # 正規化
            attmap = torch.vstack((attmap,attmap)).unsqueeze(0).unsqueeze(0) #(1,1,attmap.size(0),attmap.size(1)) にする
            map = nn.functional.interpolate(attmap,(100,256)).squeeze().squeeze() # (100,256)に変換
            
            # ECGとmapを重ねた画像を生成
            index = g_att_index.tolist().index(l) # 指定の誘導が何番目にattentionが大きいか検索
            if index < 3: # 6番目以上で
                ax[int(l%6),int(l//6)].pcolor(map,cmap=plt.cm.YlOrRd,alpha=0.2,edgecolors=None)
                ax[int(l%6),int(l//6)].imshow(ecg_)
            else:
                ax[int(l%6),int(l//6)].pcolor(map,cmap=plt.cm.GnBu,alpha=0.2,edgecolors=None)
                ax[int(l%6),int(l//6)].imshow(ecg_)
            ax[int(l%6),int(l//6)].set_title(l_name[l])
        fig.suptitle('{} True:{} Pre:{}'.format(p[i],p_name[label[i]],p_name[pred[i]]),fontweight ="bold",fontname="MS Gothic")
        plt.savefig('{}/{}/{}.png'.format(save_dir,eval,p[i]))
        plt.clf()
        plt.close()
    return

def create_confusion_matrix(cm, labels,acc,f1_score_,report,eval,save_dir):
    sns.set()
    
    df = pd.DataFrame(cm)
    df.index = labels
    df.columns = labels
    
    report_df = pd.DataFrame(report)
    report_df.to_csv('{}/report_{}.csv'.format(save_dir,eval))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df, annot=True, fmt="d", annot_kws={'size': 15},linewidths=.5, ax=ax, cmap="YlGnBu",robust=True)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('acc {:.3f}, f1 score {:.3f}'.format(acc,f1_score_))
    plt.savefig('{}/cm_{}.png'.format(save_dir,eval))
    plt.close()

def create_confusion_matrix_bi(cm, labels,acc,recall,precision,report,eval,save_dir):
    sns.set()
    
    df = pd.DataFrame(cm)
    df.index = labels
    df.columns = labels
    
    report_df = pd.DataFrame(report)
    report_df.to_csv('{}/report_{}.csv'.format(save_dir,eval))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df, annot=True, fmt="d", annot_kws={'size': 15},linewidths=.5, ax=ax, cmap="YlGnBu",robust=True)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('Accuracy {:.3f}, Recall {:.3f}, Precision {:.3f}'.format(acc,recall,precision))
    plt.savefig('{}/cm_{}.png'.format(save_dir,eval))
    plt.close()


