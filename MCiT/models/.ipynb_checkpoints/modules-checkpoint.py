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

# train for classification 
def train_model(s_model, t_model, train_loader, test_loader,num_epochs,optimizer,loss_fn,save_dir):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    s_model.to(device)
    t_model.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    #torch.backends.cudnn.benchmark = True

    history = []
    test_loss_min = np.Inf

    # epochのループ
    for epoch in range(num_epochs):
        train_loss = 0 # all loss
        train_kd_loss = 0 # kd loss
        test_loss = 0
        test_kd_loss = 0 
        train_acc = 0
        test_acc = 0

        for i,(ecg,l,p,num) in enumerate(train_loader):

            ecg = ecg.to(device)
            l = l.to(device)
            
            outputs, embed = s_model(ecg)
            # クラス識別損失
            loss = loss_fn.ce(outputs, l) * outputs.size(0)
            # kd損失
            
            kd_loss = 0
            for j in range(ecg.size(1)):
                with torch.no_grad():
                    out_t_model = t_model(ecg[:,j,:,:,:],kd=True)
                    #print(embed[:,j+1,:])
                kd_loss += loss_fn.kd(embed[:,j+1,:], out_t_model)
                
            train_loss += loss.detach().cpu().numpy().item() 
            train_kd_loss += kd_loss.detach().cpu().numpy().item()
            
            loss = loss + kd_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy by finding max log probability
            _, pred = torch.max(outputs,dim=1) 
            train_acc += torch.sum(pred == l.data)

        # epochごとのlossと正解率
        train_acc = train_acc.double() / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)
        train_kd_loss = train_kd_loss / len(train_loader.dataset)

        with torch.no_grad():
            s_model.eval()
            for ecg,l,p,num in test_loader:
                ecg = ecg.to(device)
                l = l.to(device)
                
                outputs, embed = s_model(ecg)
                # クラス識別損失
                loss = loss_fn.ce(outputs, l) * outputs.size(0)
                # kd損失
                kd_loss = 0
                for j in range(ecg.size(1)):
                    out_t_model = t_model(ecg[:,j,:,:,:],kd=True)
                    kd_loss += loss_fn.kd(embed[:,j+1,:], out_t_model)
                test_loss += loss.detach().cpu().numpy().item()
                test_kd_loss += kd_loss.detach().cpu().numpy().item()
                
                loss = loss + kd_loss

                _, pred = torch.max(outputs,dim=1) 
                test_acc += torch.sum(pred == l.data)
        
        test_loss = test_loss / len(test_loader.dataset)
        test_kd_loss = test_kd_loss / len(test_loader.dataset)
        test_acc = test_acc / len(test_loader.dataset)

        if test_loss < test_loss_min:
            test_loss_min = test_loss
            best_epoch = epoch
            best_acc = test_acc
            early_model = copy.deepcopy(s_model)
            torch.save(early_model.state_dict(), os.path.join(save_dir, 'vit_earlystoped.pth'))
            print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {test_loss_min:.2f} and acc: {100 * best_acc:.2f}%')
                
        print('Epoch {}/{} | train/test Loss: {:.4f}/{:.4f} Acc: {:.4f}/{:.4f}'.format(epoch+1, num_epochs,train_loss,test_loss,train_acc,test_acc))

        print('kd loss train/test : {:.4f} / {:.4f}'.format(train_kd_loss,test_kd_loss))
        history.append([train_loss,test_loss,train_kd_loss,test_kd_loss,train_acc.detach().cpu().item(), test_acc.detach().cpu().item()])
 
    history = np.array(history)
    save_history(history,save_dir)
    torch.save(s_model.state_dict(), os.path.join(save_dir, 'vit.pth'))
    return s_model, history

def save_history(his_loss,save_dir,mim=False):
    np.save(os.path.join(save_dir,'his_loss'), his_loss)
    if mim == True:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.plot(his_loss[:,0],label='train_loss')
        plt.legend()
        plt.title('loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('{}/loss.png'.format(save_dir))
        plt.close()

    else:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.plot(his_loss[:,0],label='train_loss')
        plt.plot(his_loss[:,1],label='test_loss')
        plt.legend()
        plt.title('loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('{}/loss.png'.format(save_dir))
        plt.close()
        his_acc = his_loss[:,[2,3]]
        plt.plot(his_acc[:,0],label='train_acc')
        plt.plot(his_acc[:,1],label='test_acc')
        plt.legend(loc='lower right')
        plt.title('accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig('{}/accuracy.png'.format(save_dir))
        plt.close()

# TE用
def classifier_accuracy(config,vit_early,vit_trained,train_dataloader,test_dataloader,save_dir,label_name,task):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 訓練終了時の正解率(acc)とF1-scoreを計算
    train_acc,train_f1_score, train_report = accuracy(device,vit_trained,train_dataloader,'train',save_dir, label_name,config,task)
    test_acc,test_f1_score, test_report = accuracy(device,vit_trained,test_dataloader,'test',save_dir,label_name,config,task)
    print('train : acc {:.3f} f1_score {:.3f}'.format(train_acc,train_f1_score))
    print('test : acc {:.3f} f1_score {:.3f}'.format(test_acc,test_f1_score))

    # テストに対する最小loss時の正解率(acc)とF1-scoreを計算
    vit_early = vit_early.to(device)
    vit_early.load_state_dict(torch.load(os.path.join(save_dir, 'vit_earlystoped.pth')))
    e_train_acc,e_train_f1_score, e_train_report = accuracy(device,vit_early,train_dataloader,'early_train',save_dir,label_name,config,task)
    e_test_acc,e_test_f1_score, e_test_report = accuracy(device,vit_early, test_dataloader,'early_test',save_dir,label_name,config,task)
    print('early_train : acc {:.3f} f1_score {:.3f}'.format(e_train_acc,e_train_f1_score))
    print('early_test : acc {:.3f} f1_score {:.3f}'.format(e_test_acc,e_test_f1_score))

    # 部位識別に対する訓練acc,訓練f値, テストacc,テストf値の4つと, ACS識別(2class)に対する訓練acc,訓練f値, テストacc,テストf値の4つをまとめて返す
    return e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score

def accuracy(device,model_,loader,eval,save_dir,label_name,config,task,attention_show_flg=False):
    preds = np.array([]) #予測ラベル
    targets = np.array([]) #正解ラベル
    patients = np.array([]) #患者番号
    result_history = np.array([]) #誤診断した患者番号
    attention_leads = np.array([]) # 各誘導におけるAttention可視化
    attention_leads_index = np.array([]) # Attentionが大きかった誘導indexを保存
    num_leads = config.model_conf.num_lead * config.model_conf.patch_num_v * config.model_conf.patch_num_h + 1
    with torch.no_grad():
        model_.eval()
        #corrects=0
        for ecg, label,p,num in loader:
            ecg = ecg.to(device)
            l = label.to(device)
            
            outputs, embd  = model_(ecg,attention_show_flg=attention_show_flg)
            if attention_show_flg == True:
                attention_leads_index=np.append(attention_leads_index,g_att.argsort()[:,:,::-1])

            _, pred = torch.max(outputs, 1)  # ラベルを予測

            preds = np.append(preds, pred.cpu().data.numpy())
            targets = np.append(targets,label.cpu().data.numpy())
            patients = np.append(patients,p)
        
        for i in range(len(targets)):
            if targets[i] == preds[i]:
                result_history=np.append(result_history,[patients[i],'0',targets[i],preds[i]])
            else:
                result_history=np.append(result_history,[patients[i],'1',targets[i],preds[i]])
        result_history =  result_history.reshape(-1,4)
        pd.DataFrame(result_history,columns=['patients','result','target','prediction']).to_csv('{}/{}_miss_patients.csv'.format(save_dir,eval))
        test_acc = accuracy_score(targets,preds)
        f1_score_ = f1_score(targets, preds,average='macro')
        report = classification_report(targets,preds,output_dict=True)
        
        cm = confusion_matrix(targets, preds)
        create_confusion_matrix(cm,label_name,test_acc,f1_score_,report,eval,save_dir)
        
        return test_acc, f1_score_, report


def create_confusion_matrix(cm, labels,acc,f1_score_,report,eval,save_dir):
    sns.set()
    
    df = pd.DataFrame(cm)
    df.index = labels
    df.columns = labels
    df.to_csv('{}/confusion_matrix_{}.csv'.format(save_dir,eval))
    
    report_df = pd.DataFrame(report)
    report_df.to_csv('{}/report_{}.csv'.format(save_dir,eval))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df, annot=True, fmt="d", annot_kws={'size': 15},linewidths=.5, ax=ax, cmap="YlGnBu",robust=True)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('acc {:.3f}, f1 score {:.3f}'.format(acc,f1_score_))
    plt.savefig('{}/cm_{}.png'.format(save_dir,eval))
    plt.close()


