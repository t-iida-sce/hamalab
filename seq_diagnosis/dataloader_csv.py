import numpy as np
import random
import matplotlib.pyplot as plt
import os
import glob
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as tf
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import yaml

from torch.utils.data import TensorDataset, DataLoader, Subset 
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageOps
import pandas as pd

# class CustomTensorDataset(TensorDataset):
#     """TensorDataset with support of transforms.
#     """
#     def __init__(self, tensors,labels,p,num,transform):
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#         print(tensors.size())
#         self.tensors = tensors
#         self.labels = labels
#         self.p = p
#         self.num = num
#         self.transform = transform
#         print(len(p))
#         print(len(num))

#     def __getitem__(self, index):
#         if self.transform == False:
#             tuple1 = tuple(tensor[index] for tensor in self.tensors) 
#             return tuple1   
#         tuple1 = torch.stack(tuple(self.transform(tensor[index]) for tensor in self.tensors))
#         label = self.labels[index]
#         p = self.p[index]
#         num = self.num[index]
#         return tuple1,label,p,num

#     def __len__(self):
#         return self.tensors[0].size(0)

class CustomTensorDataset_csv(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors,labels,transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        print(tensors.size())
        self.tensors = tensors
        self.labels = labels
        self.transform = transform
        print(len(labels))

    def __getitem__(self, index):
        # if self.transform == False:
        #     #   tuple1 = tuple(tensor[index] for tensor in self.tensors) 
        #     tuple1 = self.tensors[index]
        #     return tuple1 
        
        if self.transform is not None:
            #tuple1 = torch.stack(tuple(self.transform(tensor[index]) for tensor in self.tensors))  #テンソルの中の1データ
            tuple1 = torch.stack(tuple(self.transform(self.tensors[index])))
        else:
            tuple1=self.tensors[index]
        label = self.labels[index]
        return tuple1,label

    def __len__(self):
        return self.tensors.size(0)
    


# 時系列用データローダー
def create_dataloader_csv(data_dir,batch_size,seed=None):
    ecg_seq=[] # データ 
    labels=[] # ラベル
    transform = transforms.Compose([])
    
    # ---------- get all dataset ----------  ラベル抜き出し、数字と結びつける
    label_=[]
    for f in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, f)):
            label_.append(f)
    label_to_idx = np.arange(len(label_)).tolist()
    label_dict = dict(zip(label_,label_to_idx))
    print('Classification label :',label_dict)

    # create dataloader 
    ecg_seq = []
    labels=[] # ラベル
    transform = transforms.Compose([])
    for label_name in label_:
        label_dir = os.path.join(data_dir,label_name,'*')
        for path in glob.glob(label_dir):
            if path[-3:] == 'ini':
                continue
            # CSVファイルを読み込む
            df = pd.read_csv(path)
            # 各列のデータを配列に格納する
            seq1 = df['V6']
            seq2 = df['a_‡T']
            seq3 = df['a_‡U']
            seq4 = df['a_‡V']
            seq5 = df['a_aVR']
            seq6 = df['a_aVL']
            seq7 = df['a_aVF']
            seq8 = df['a_V1']
            seq9 = df['a_V2']
            seq10 = df['a_V3']
            seq11 = df['a_V4']
            seq12 = df['a_V5']

            seq1 = torch.tensor(seq1)
            seq2 = torch.tensor(seq2)
            seq3 = torch.tensor(seq3)
            seq4 = torch.tensor(seq4)
            seq5 = torch.tensor(seq5)
            seq6 = torch.tensor(seq6)
            seq7 = torch.tensor(seq7)
            seq8 = torch.tensor(seq8)
            seq9 = torch.tensor(seq9)
            seq10 = torch.tensor(seq10)
            seq11 = torch.tensor(seq11)
            seq12 = torch.tensor(seq12)

            seq = torch.stack([seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9, seq10, seq11, seq12])
            ecg_seq.append(seq)
            labels.append(label_dict[label_name])
    labels = torch.tensor(labels)
    
    ecg_seq = torch.stack(ecg_seq, dim = 0)
    # ecg_seq = torch.transpose(ecg_seq, 1, 2)
    
    dataset = CustomTensorDataset_csv(ecg_seq,labels,transform)
    


    # TensorDatasetの作成
    #dataset = TensorDataset(ecg_seq, labels)   

    print("test1")
    print('dataset len :', len(dataset))
    print('data 0:',dataset[0])
    print('labels len :', len(labels))
    print(labels)
    print("test2")

    # ---------- create data loader ----------

    TEST_SIZE = 0.2
    if seed == None:
        seed = random.randint(1,100)

    # # generate indices: instead of the actual data we pass in integers instead
    # train_indices, test_indices, _, _= train_test_split(
    #     range(len(dataset)),
    #     dataset.labels,
    #     stratify=dataset.labels,
    #     test_size=TEST_SIZE,
    #     random_state=seed
    # )

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=12)

    print("indices")
    print(len(train_indices))
    print(len(test_indices))


    # generate subset based on indices
    train_split = Subset(dataset, train_indices)
    test_split = Subset(dataset, test_indices)
    print(train_split)
    # create batches
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_split, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader,labels
