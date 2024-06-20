import torch
import torch.nn as nn
import torch.nn.functional as F


class kd_loss_kl:
    def __init__(self,config):
        self.kd_model = config.teacher_model
        self.alpha = config.alpha
        self.temp = config.temperature
    
    def ce(self, outputs, labels):
        if self.kd_model:
            return (1-self.alpha) * nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        else:
            return nn.CrossEntropyLoss(reduction='mean')(outputs, labels)

    def kd(self, outputs, teacher_outputs):
        #print(F.log_softmax(outputs/self.temp, dim=1))
        #print(F.log_softmax(teacher_outputs/self.temp, dim=1))
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/self.temp, dim=1),F.softmax(teacher_outputs/self.temp, dim=1)) * (self.alpha * self.temp * self.temp)
        return  kd_loss

class kd_loss_l2:
    def __init__(self,config):
        self.kd_model = config.teacher_model
        self.alpha = config.alpha
    
    def ce(self, outputs, labels):
        if self.kd_model:
            #return (1-self.alpha) * nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
            return nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        else:
            return nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
    def kd(self, outputs, teacher_outputs):
        #l2loss = torch.nn.MSELoss(reduction='mean')(outputs, teacher_outputs) * self.alpha
        l2loss = torch.nn.MSELoss(reduction='mean')(outputs, teacher_outputs)
        return l2loss