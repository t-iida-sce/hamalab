import os
import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score, precision_score,classification_report
import torchvision.models as models
import torchvision.transforms as transforms
import torch

from models.mlp_head import MLPHead
from models import resnet_attention

class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        if config.name == 'CNN6':
            self.encoder = CNN6()
            self.in_features = 256
        elif config.name == 'resnet18':
            resnet = resnet_attention.resnet18(pretrained=False)
            self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.in_features = resnet.fc.in_features
        elif config.name == 'resnet18s':
            resnet = resnet_attention.resnet18s(pretrained=False)
            self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.in_features = resnet.fc.in_features
        elif config.name == 'axial18':
            axialnet = resnet_attention.axial18s(pretrained=False)
            self.encoder = axialnet
            self.in_features = axialnet.fc.in_features
        
        if config.cls==1:
            self.projection = torch.nn.Linear(self.in_features, config.projection_head)
        else:
            self.projection= MLPHead(in_channels=self.in_features, projection_size=config.projection_head)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

class Net26(torch.nn.Module):
    def __init__(self, *args, **config):
        super(Net26, self).__init__()
        if config.name == 'resnet26':
            resnet = resnet_attention.resnet26(pretrained=False)
            self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.in_features = resnet.fc.in_features
            self.projection = MLPHead(in_channels=self.in_features, projection_size=config.projection_head)
        elif config.name == 'axial26':
            axialnet = resnet_attention.axial26s(pretrained=False)
            self.encoder = axialnet
            self.in_features = axialnet.fc.in_features
            self.projection = MLPHead(in_channels=self.in_features, projection_size=config.projection_head)


    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class Net50(torch.nn.Module):
    def __init__(self, *args, **config):
        super(Net50, self).__init__()
        if config.name == 'resnet50':
            resnet = models.resnet50(pretrained=False)
            self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.in_features = resnet.fc.in_features
            self.projection = MLPHead(in_channels=self.in_features, projection_size=config.projection_head)
        elif config.name == 'axial50':
            axialnet = resnet_attention.axial50s(pretrained=False)
            self.encoder = axialnet
            self.in_features = axialnet.fc.in_features
            self.projection = MLPHead(in_channels=self.in_features, projection_size=config.projection_head)
        elif config.name == 'axial50_2':
            axialnet = resnet_attention.axial50s(pretrained=False)
            self.encoder = torch.nn.Sequential(*list(axialnet.children())[:-1])
            self.in_features = axialnet.fc.in_features
            self.projection = MLPHead(in_channels=self.in_features, projection_size=config.projection_head)


    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

class CNN6(torch.nn.Module):
    def __init__(self):
        super(CNN6, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128,256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = torch.nn.AdaptiveMaxPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.maxpool(x)
        return x    
    
# trainer
def train_model(model, train_loader, test_loader,num_epochs,optimizer,criterion,save_dir,num_lead=12):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    model.to(device)

    if num_lead==12:
        h_num = 2
    else:
        h_num = 3
    
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
            ecg = ecg.permute(0,2,1,3,4)
            inp = ecg.reshape(ecg.size(0),ecg.size(1),-1,ecg.size(4)*h_num)
            inp = inp.to(device)
            l = l.to(device)
            outputs = model(inp)

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
                ecg = ecg.permute(0,2,1,3,4)
                inp = ecg.reshape(ecg.size(0),ecg.size(1),-1,ecg.size(4)*h_num)
                inp = inp.to(device)
                l = l.to(device)
                outputs = model(inp)

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
            torch.save(early_model.state_dict(), os.path.join(save_dir, 'cnn_earlystopped.pth'))
            print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {test_loss_min:.2f} and acc: {100 * best_acc:.2f}%')
                
        print('Epoch {}/{} | train/test Loss: {:.4f}/{:.4f} Acc: {:.4f}/{:.4f}'.format(epoch+1, num_epochs,train_loss,test_loss,train_acc,test_acc))

        history.append([train_loss,test_loss,train_acc.detach().cpu().item(), test_acc.detach().cpu().item()])

    history = np.array(history)
    save_history(history,save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'cnn.pth'))
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

def classifier_accuracy(config,vit_early,vit_trained,train_dataloader,test_dataloader,save_dir,label_name,num_lead=12,val_dataloader=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 訓練終了時の正解率(acc)とF1-scoreを計算
    train_acc,train_f1_score, train_report = accuracy(device,vit_trained,train_dataloader,'train',save_dir, label_name,num_lead)
    if val_dataloader != False:
        val_acc, val_f1_score, val_report = accuracy(device,vit_trained,val_dataloader,'val',save_dir,label_name,num_lead)
    test_acc,test_f1_score, test_report = accuracy(device,vit_trained,test_dataloader,'test',save_dir,label_name,num_lead)
    print('train : acc {:.3f} f1_score {:.3f}'.format(train_acc,train_f1_score))
    print('test : acc {:.3f} f1_score {:.3f}'.format(test_acc,test_f1_score))

    # テストに対する最小loss時の正解率(acc)とF1-scoreを計算
    vit_early = vit_early.to(device)
    vit_early.load_state_dict(torch.load(os.path.join(save_dir, 'cnn_earlystopped.pth')))
    e_train_acc,e_train_f1_score, e_train_report = accuracy(device,vit_early,train_dataloader,'early_train',save_dir,label_name,num_lead)
    if val_dataloader != False:
        e_val_acc,e_val_f1_score, e_val_report = accuracy(device,vit_early, val_dataloader,'early_val',save_dir,label_name,num_lead)
    e_test_acc,e_test_f1_score, e_test_report = accuracy(device,vit_early, test_dataloader,'early_test',save_dir,label_name,num_lead)
    print('early_train : acc {:.3f} f1_score {:.3f}'.format(e_train_acc,e_train_f1_score))
    print('early_test : acc {:.3f} f1_score {:.3f}'.format(e_test_acc,e_test_f1_score))

    if val_dataloader != False:
        return e_train_acc,e_train_f1_score,e_val_acc,e_val_f1_score,e_test_acc,e_test_f1_score
    else:
        return e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score

def accuracy(device,model,loader,eval,save_dir,label_name,num_lead,attention_show_flg=False):
    if num_lead==12:
        h_num = 2
    else:
        h_num = 3
        
    preds = np.array([]) #予測ラベル
    targets = np.array([]) #正解ラベル
    patients = np.array([]) #患者番号
    result_history = np.array([]) #誤診断した患者番号

    with torch.no_grad():
        model.eval()
        #corrects=0
        for ecg, l,p,num in loader:
            ecg = ecg.permute(0,2,1,3,4)
            inp = ecg.reshape(ecg.size(0),ecg.size(1),-1,ecg.size(4)*h_num)
            inp = inp.to(device)
            l = l.to(device)
            outputs = model(inp)

            _, pred = torch.max(outputs, 1)  # ラベルを予測

            preds = np.append(preds, pred.cpu().data.numpy())
            targets = np.append(targets,l.cpu().data.numpy())
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
