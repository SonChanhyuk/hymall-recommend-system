import torch
import torch.nn as nn # 신경망들이 포함됨
from torch.nn import functional as F
import torch.nn.init as init # 텐서에 초기값을 줌
from torch.utils.data.sampler import Sampler
import numpy as np

class MPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, m=4):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.m = m
        
    def __len__(self):
        return len(self.labels) // self.batch_size
    
    def __iter__(self):
        for _ in range(self.__len__()):
            labels_in_batch = set()
            idx = np.array([], dtype=np.int)

            while idx.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if (sample_label in labels_in_batch):
                    continue
                sample_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                if (len(sample_ids) < self.m):
                    continue
                labels_in_batch.add(sample_label)
                subsample = np.random.permutation(sample_ids)[:self.m]
                idx = np.append(idx, subsample)
            idx = idx[:self.batch_size]
            idx = np.random.permutation(idx)
            yield list(idx)

resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

class GlobalDescriptor(nn.Module):
    def __init__(self,pk=3):
        super().__init__()
        self.pk = pk
    
    def forward(self, x):
        if self.pk == 1: #SPoC
            return x.mean(dim=[-1,-2])
        elif self.pk == float('inf'): #MAC
            return torch.flatten(F.adaptive_max_pool2d(x,output_size=(1,1)), start_dim=1)
        else: #GeM
            sum_x = x.pow(self.pk).mean(dim=[-1,-2])
            return torch.sign(sum_x) * torch.abs(sum_x).pow(1.0/self.pk)

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.normalize(x,p=2,dim=-1)

class CGD(nn.Module):
    def __init__(self, backbone_type, gd_config, feature_dim, num_classes):
        super().__init__()
        
        # backbone model에서 AvgPool2d부분과 Linear부분을 제거하여 output을 줄이지 않도록하여 네트워크를 구성
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.features = []
        for name, module in backbone.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)
        
        # Main Module : Multiple Global Descriptors
        n = len(gd_config)
        k = feature_dim // n
        assert feature_dim % n == 0, 'the feature dim should be divided by number of global descriptors'
        
        self.global_descriptors, self.main_modules = [],[]
        for i in range(n):
            if gd_config[i] == "S":
                pk = 1
            elif gd_config[i] == "M":
                pk = float('inf')
            else:
                pk = 3
            self.global_descriptors.append(GlobalDescriptor(pk))
            self.main_modules.append(nn.Sequential(nn.Linear(2048, k, bias=False), L2Norm()))
        self.global_descriptors = nn.ModuleList(self.global_descriptors)
        self.main_modules = nn.ModuleList(self.main_modules)
        
        # Auxiliary Module : Classification Loss
        self.auxiliary_module = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, num_classes, bias=True))
        
    def forward(self, x):
        shared = self.features(x.float())
        global_descriptors = []
        for i in range(len(self.global_descriptors)):
            global_descriptor = self.global_descriptors[i](shared)
            if i == 0:
                classes = self.auxiliary_module(global_descriptor)
            global_descriptor = self.main_modules[i](global_descriptor)
            global_descriptors.append(global_descriptor)
        global_descriptors = F.normalize(torch.cat(global_descriptors, dim=-1),dim=-1)
        return global_descriptors, classes

class CustomTripletMarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
    
    @staticmethod
    def anchor_positive(label): #target내의 같은 label위치를 True로
        mask = torch.eq(label.unsqueeze(0), label.unsqueeze(1))
        mask.fill_diagonal_(False)
        return mask
        
    @staticmethod
    def anchor_negative(label):
        labels_equal = torch.eq(label.unsqueeze(0), label.unsqueeze(1))
        mask = ~ labels_equal
        return mask
    
    def forward(self,feature,label):
        pairwise_dist = torch.cdist(feature.unsqueeze(0), feature.unsqueeze(0)).squeeze(0)

        anchor_positive = self.anchor_positive(label)
        anchor_positive_dist = anchor_positive.float() * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        anchor_negative = self.anchor_negative(label)
        
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - anchor_negative.float())
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        loss = (F.relu(hardest_positive_dist - hardest_negative_dist + self.margin))
        return loss.mean()

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
