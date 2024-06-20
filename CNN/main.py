import os
from pathlib import Path
import yaml
import codecs
import random
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from data.create_dataset_dummy import create_dataset4,create_dataset6,create_dataset9, create_dataset12
from data.create_dataset import create_dataloader, create_dataloader_18,create_dataloader_only12

from torch.utils.data import DataLoader, Subset 
from models import resnet_base_network_all, resnet_base_network_each, MVCNN, MHBN, GVCNN, MFB_

def get_dataset(config,seed):
    batch_size = config.model_conf.batch_size
    input_size = config.model_conf.input_size
    task = config.dataset.task
    num_view = config.model_conf.num_view
    input_random = config.dataset.input_random
    da = config.dataset.data_augumentation
    
    if task == 2:
        data_dir = '{}/dataset/ECG100_224_2class/'.format(get_original_cwd())
        if num_view == 12:
            train_dataloader, test_dataloader, label_name, train_len, test_len = create_dataloader_only12(data_dir, batch_size,input_size,seed,input_random,da=da)
        elif num_view == 18:
            train_dataloader, test_dataloader, label_name, train_len, test_len = create_dataloader_18(data_dir, batch_size,input_size,seed,input_random,da=da)
        print('Data Augumentation : ', da)

    elif task == 4:
        data_dir = '{}/dataset/ECG100_224_ACS/'.format(get_original_cwd())
        if num_view == 12:
            train_dataloader, test_dataloader, label_name, train_len, test_len = create_dataloader_only12(data_dir, batch_size,input_size,seed,input_random,da=da)
        elif num_view == 18:
            train_dataloader, test_dataloader, label_name, train_len, test_len = create_dataloader_18(data_dir, batch_size,input_size,seed,input_random,da=da)
        print('Data Augumentation : ', da)

    elif task == 5:
        data_dir = '{}/dataset/ECG100_224/'.format(get_original_cwd())
        if num_view == 12:
            train_dataloader, test_dataloader, label_name, train_len, test_len = create_dataloader_only12(data_dir, batch_size,input_size,seed,input_random,da=da)
        elif num_view == 18:
            train_dataloader, test_dataloader, label_name, train_len, test_len = create_dataloader_18(data_dir, batch_size,input_size,seed,input_random,da=da)
        print('Data Augumentation : ', da)

    return train_dataloader, test_dataloader, label_name

def exp(config,savepath):

    score = np.array([])
    seeds = [0,25,50,75,100]
    for i,seed in enumerate(seeds):
        # Get dataset
        train_dataloader, test_dataloader, label_name = get_dataset(config,seed)

        # Classifier task
        savepath_times = os.path.join(savepath,'{}/'.format(i))
        print(savepath_times)
        if not os.path.exists(savepath_times):
            os.mkdir(savepath_times)
        classifier_score = classifier(config,train_dataloader, test_dataloader,savepath_times, label_name)

        score = np.append(score,classifier_score)

    summaryscore_path = os.path.join(savepath,'ealry_score.csv')
    score = score.reshape([-1,4])
    score=pd.DataFrame(score)
    score.columns = ['train_acc','train_f1','test_acc','test_f1']
    score.to_csv(summaryscore_path) # scoreの保存

    return 

def exp_dummy(config,savepath):
    type_ = config.type
    batch_size = config.model_conf.batch_size
    input_size = config.model_conf.input_size
    
    if config.dataset.data == 0:
        train_data_dir = '{}/dataset/low_freq_data/train/'.format(get_original_cwd())
        test_data_dir = '{}/dataset/low_freq_data/test/'.format(get_original_cwd())
        dataset, label_name = create_dataset9(train_data_dir, batch_size,input_size,da=config.dataset.data_augumentation)
        test_dataset, _= create_dataset9(test_data_dir, batch_size,input_size,da=config.dataset.data_augumentation)
    if config.dataset.data == 1:
        train_data_dir = '{}/dataset/shape_data/train/'.format(get_original_cwd())
        test_data_dir = '{}/dataset/shape_data/test/'.format(get_original_cwd())
        dataset, label_name = create_dataset4(train_data_dir, batch_size,input_size,da=config.dataset.data_augumentation)
        test_dataset, _= create_dataset4(test_data_dir, batch_size,input_size,da=config.dataset.data_augumentation)
    if config.dataset.data == 2:
        train_data_dir = '{}/dataset/low_freq_datav2/train/'.format(get_original_cwd())
        test_data_dir = '{}/dataset/low_freq_datav2/test/'.format(get_original_cwd())
        dataset, label_name = create_dataset6(train_data_dir, batch_size,input_size,da=config.dataset.data_augumentation)
        test_dataset, _ = create_dataset6(test_data_dir, batch_size,input_size,da=config.dataset.data_augumentation)
    
    print('Data Augumentation : ', config.dataset.data_augumentation)
        
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    kf = KFold(n_splits=5,shuffle=True,random_state=0)

    score = np.array([])
    
    for _fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_index)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dataset   = Subset(dataset, val_index)
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

        # Classifier task
        savepath_times = os.path.join(savepath,'{}/'.format(_fold))
        print(savepath_times)
        if not os.path.exists(savepath_times):
            os.mkdir(savepath_times)
        # Classifier task
        classifier_score = classifier(config,train_dataloader, test_dataloader, savepath_times, label_name, valid_dataloader=valid_dataloader)

        score = np.append(score,classifier_score)

    summaryscore_path = os.path.join(savepath,'ealry_score.csv')
    score = score.reshape([-1,6])
    score=pd.DataFrame(score)
    score.columns = ['train_acc','train_f1','val_acc','val_f1','test_acc','test_f1']
    score.to_csv(summaryscore_path) # scoreの保存

    return 

def classifier(config,train_dataloader, test_dataloader, save_path, label_name, valid_dataloader=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if config.type == 'each':
        if config.model_conf.model == 'MVCNN':
            net = MVCNN.MVCNN(config.model_conf,device).to(device)
        elif config.model_conf.model == 'MVCNNNET':
            net = MVCNN.MVCNN_Net(config.model_conf,device).to(device)
        elif config.model_conf.model == 'CNNNET_flatten':
            net = resnet_base_network_each.Net(config.model_conf,device).to(device)
        elif config.model_conf.model == 'CNN_flatten':
            net = resnet_base_network_each.CNN(config.model_conf,device).to(device)
        elif config.model_conf.model == 'GVCNN':
            net = GVCNN.GVCNN(config.model_conf).to(device)
        elif config.model_conf.model == 'MHBN':
            net = MHBN.MHBNN(config.model_conf).to(device)
        elif config.model_conf.model == 'MFB':
            net = MFB.MFB(config.model_conf,device).to(device)
        elif config.model_conf.model == 'MFB_one':
            net = MFB.MFB_one(config.model_conf,device).to(device)
    
        print('Number of parameters : ',sum(p.numel() for p in net.parameters()))

        for name, param in net.named_parameters():
            param.requires_grad = True

        # 最適化手法の設定
        optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': config.lr}, # 'params': net.te.encoder.layer[config.classifier_conf.parameter_layer].parameters(), 'lr': config.classifier_conf.enc_lr}
        ], betas=(0.9, 0.999))

        criterion = torch.nn.CrossEntropyLoss()

        if True:
            print('更新パラメータの確認')
            for name, param in net.named_parameters():
                print(name,param.requires_grad)

        # 学習
        classifier_savepath = os.path.join(save_path,'classifier_train')
        if not os.path.exists(classifier_savepath):
            os.mkdir(classifier_savepath)

        if valid_dataloader==False:
            model_trained, history = resnet_base_network_each.train_model(net,train_dataloader,test_dataloader,config.epochs,optimizer,criterion,classifier_savepath)
            e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score = resnet_base_network_each.classifier_accuracy(config,net,model_trained,train_dataloader,test_dataloader,classifier_savepath,label_name)

            return [e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score]
        else:
            model_trained, history = resnet_base_network_each.train_model(net,train_dataloader,test_dataloader,config.epochs,optimizer,criterion,classifier_savepath)
            e_train_acc,e_train_f1_score,e_val_acc,e_val_f1_score,e_test_acc,e_test_f1_score = resnet_base_network_each.classifier_accuracy(config,net,model_trained,train_dataloader,test_dataloader,classifier_savepath,label_name,val_dataloader=valid_dataloader)

            return [e_train_acc,e_train_f1_score,e_val_acc,e_val_f1_score,e_test_acc,e_test_f1_score]
    
    else:
        net = resnet_base_network_all.Net(config.model_conf).to(device)
        print('Number of parameters : ',sum(p.numel() for p in net.parameters()))

        for name, param in net.named_parameters():
            param.requires_grad = True

        # 最適化手法の設定
        optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': config.lr}, # 'params': net.te.encoder.layer[config.classifier_conf.parameter_layer].parameters(), 'lr': config.classifier_conf.enc_lr}
        ], betas=(0.9, 0.999))

        criterion = torch.nn.CrossEntropyLoss()

        if True:
            print('更新パラメータの確認')
            for name, param in net.named_parameters():
                print(name,param.requires_grad)

        # 学習
        classifier_savepath = os.path.join(save_path,'classifier_train')
        if not os.path.exists(classifier_savepath):
            os.mkdir(classifier_savepath)

        if valid_dataloader==False:
            model_trained, history = resnet_base_network_all.train_model(net,train_dataloader,test_dataloader,config.epochs,optimizer,criterion,classifier_savepath,config.model_conf.num_view)
            e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score = resnet_base_network_all.classifier_accuracy(config,net,model_trained,train_dataloader,test_dataloader,classifier_savepath,label_name,config.model_conf.num_view)

            return [e_train_acc,e_train_f1_score,e_test_acc,e_test_f1_score]
        else:
            model_trained, history = resnet_base_network_all.train_model(net,train_dataloader,test_dataloader,config.epochs,optimizer,criterion,classifier_savepath,config.model_conf.num_view)
            e_train_acc,e_train_f1_score,e_val_acc,e_val_f1_score,e_test_acc,e_test_f1_score = resnet_base_network_all.classifier_accuracy(config,net,model_trained,train_dataloader,test_dataloader,classifier_savepath,label_name,config.model_conf.num_view,val_dataloader=valid_dataloader)

            return [e_train_acc,e_train_f1_score,e_val_acc,e_val_f1_score,e_test_acc,e_test_f1_score]


@hydra.main(config_path="defaults.yaml")
def main(config: DictConfig):
    #base_config = OmegaConf.structured(config)
    #cli_config = OmegaConf.from_cli()
    #config = OmegaConf.merge(base_config, cli_config)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    cwd_path = get_original_cwd()

    print(config)
    
    cls =['MLP','Linear','pool','poolmlp']
    if config.dataset.dummy != False:
        data_name = ['lowfreq','shape','lowfreq2']
        savepath = '{}/runs/{}/{}_{}/{}/{}/'.format(cwd_path,data_name[config.dataset.data],config.model_conf.model,config.model_conf.name,config.type,cls[config.model_conf.cls]) # 保存先 
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        config_path = savepath+'/config.yaml'
        OmegaConf.save(config, config_path)
        print(savepath)
        
        exp_dummy(config,savepath)
        
    else:
        input_order = ['in_asc','in_rand']
        savepath = '{}/runs/view{}_{}_{}/{}_{}/{}/{}/'.format(cwd_path,config.model_conf.num_view,config.dataset.task,input_order[config.dataset.input_random],config.model_conf.model,config.model_conf.name,config.type,cls[config.model_conf.cls]) # 保存先 
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        config_path = savepath+'/config.yaml'
        OmegaConf.save(config, config_path)
        print(savepath)
    
        exp(config,savepath)

if __name__ == '__main__':
    main()
    