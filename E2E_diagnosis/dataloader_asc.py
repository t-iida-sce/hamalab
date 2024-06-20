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

class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors,labels,p,num,transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        print(tensors.size())
        self.tensors = tensors
        self.labels = labels
        self.p = p
        self.num = num
        self.transform = transform
        print(len(p))
        print(len(num))

    def __getitem__(self, index):
        if self.transform == False:
            tuple1 = tuple(tensor[index] for tensor in self.tensors) 
            return tuple1   
        tuple1 = torch.stack(tuple(self.transform(tensor[index]) for tensor in self.tensors))
        label = self.labels[index]
        p = self.p[index]
        num = self.num[index]
        return tuple1,label,p,num

    def __len__(self):
        return self.tensors[0].size(0)

class CustomTensorDataset_(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors,labels,label_lead,p,num,transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        print(tensors.size())
        self.tensors = tensors
        self.labels = labels
        self.label_lead = label_lead
        self.p = p
        self.num = num
        self.transform = transform
        print(len(p))
        print(len(num))

    def __getitem__(self, index):
        if self.transform == False:
            tuple1 = tuple(tensor[index] for tensor in self.tensors) 
            return tuple1   
        tuple1 = torch.stack(tuple(self.transform(tensor[index]) for tensor in self.tensors))
        label = self.labels[index]
        label_lead = self.label_lead[index]
        p = self.p[index]
        num = self.num[index]
        return tuple1,label,label_lead,p,num

    def __len__(self):
        return self.tensors[0].size(0)
    
class CustomTensorDataset__(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors,labels,p,transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        print(tensors.size())
        self.tensors = tensors
        self.labels = labels
        self.p = p
        self.transform = transform
        print(len(p))

    def __getitem__(self, index):
        if self.transform == False:
            tuple1 = tuple(tensor[index] for tensor in self.tensors) 
            return tuple1   
        tuple1 = torch.stack(tuple(self.transform(tensor[index]) for tensor in self.tensors))
        label = self.labels[index]
        p = self.p[index]
        return tuple1,label,p

    def __len__(self):
        return self.tensors[0].size(0)


# loaderを返す
# loaderはデータ, ラベル, 患者番号, leads数を返す

def create_dataset_12_asc(data_dir,batch_size,seed=None):
    ecg_image=[] # データ
    labels=[] # ラベル
    label_lead=[]
    label_about=[]
    patients=[] # 患者番号
    num_leads=[] # leads数
    transform = transforms.Compose([])
    transform_ = transforms.Compose([transforms.Pad(padding=(0, 6),padding_mode="edge"),transforms.ToTensor()])
    # 112 * 224 に整形
    # ---------- get all dataset ----------
    label_ = []
    label_true = [0,1,2,3,4,5,6,7,8,9,10,11]
    patients_ = []
    for f in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, f)):
            label_.append(f)
            patients_dir=os.path.join(data_dir, f)
            for p in os.listdir(patients_dir):
                if os.path.isdir(os.path.join(patients_dir, p)):
                    patients_.append(p)
    label_to_idx = np.arange(len(label_)).tolist()
    label_dict = dict(zip(label_,label_to_idx))
    print('Classification lab :',label_dict)
    print('Number of datas',len(patients_))
    i,j=0,0
    # create dataloader
    for label_name in label_:
        for patients_name in patients_:
            flag = False
            label_dir = os.path.join(data_dir,label_name,patients_name,'*')
            imgs_13, imgs_17 = None,None

            for path in glob.glob(label_dir):
                if path[-3:] == 'jpg':
                    flag=True # ファイルが存在
                    img = Image.open(path)
                    img = img.convert('L').convert("RGB")
                    img = ImageOps.invert(img)
                    img= transform_(img)
                    if path[-6:-4] == '01':
                        imgs_01 = img
                    elif path[-6:-4] == '02':
                        imgs_02 = img
                    elif path[-6:-4] == '03':
                        imgs_03 = img
                    elif path[-6:-4] == '04':
                        imgs_04 = img
                    elif path[-6:-4] == '05':
                        imgs_05 = img
                    elif path[-6:-4] == '06':
                        imgs_06 = img
                    elif path[-6:-4] == '07':
                        imgs_07 = img
                    elif path[-6:-4] == '08':
                        imgs_08 = img
                    elif path[-6:-4] == '09':
                        imgs_09 = img
                    elif path[-6:-4] == '10':
                        imgs_10 = img
                    elif path[-6:-4] == '11':
                        imgs_11 = img
                    elif path[-6:-4] == '12':
                        imgs_12 = img
            if flag == True: # ファイルが存在したら加える
                asc = torch.arange(12).tolist() # 誘導番号順を乱数生成
                #print(asc)
                imgs = [imgs_01,imgs_02,imgs_03,imgs_04,imgs_05,imgs_06,imgs_07,imgs_08,imgs_09,imgs_10,imgs_11,imgs_12]
                lead_dict=dict(zip(label_true,imgs))
                image = torch.stack([lead_dict[asc[0]],lead_dict[asc[1]],lead_dict[asc[2]],lead_dict[asc[3]],lead_dict[asc[4]],lead_dict[asc[5]],
                                     lead_dict[asc[6]],lead_dict[asc[7]],lead_dict[asc[8]],lead_dict[asc[9]],lead_dict[asc[10]],lead_dict[asc[11]]])
                num_leads.append(12)
                ecg_image.append(image)
                labels.append(label_dict[label_name])
                label_lead.append(asc)
                patients.append(patients_name)
    ecg_image = pad_sequence(ecg_image)
    print("イメージ形")

    labels = torch.tensor(labels)
    label_lead = torch.tensor(label_lead)
    dataset = CustomTensorDataset_(ecg_image,labels,label_lead,patients,num_leads,transform=transform)
    # ---------- create data loader ----------

    TEST_SIZE = 0.2
    if seed == None:
        seed = random.randint(1,100)

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.labels,
        stratify=dataset.labels,
        test_size=TEST_SIZE,
        random_state=seed
    )

    # generate subset based on indices
    train_split = Subset(dataset, train_indices)
    test_split = Subset(dataset, test_indices)

    # create batches
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_split, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader,label_lead, labels