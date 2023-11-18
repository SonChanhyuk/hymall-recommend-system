import torch
import torch.nn as nn # 신경망들이 포함됨
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import CGD, CustomTripletMarginLoss, MPerClassSampler
from dataset import ClothImageDataset
from torch.utils.tensorboard import SummaryWriter

def recall(feature_vectors, feature_labels, rank):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)

    dist_matrix = torch.cdist(feature_vectors.unsqueeze(0), feature_vectors.unsqueeze(0)).squeeze(0)

    dist_matrix.fill_diagonal_(float('inf'))

    idx = dist_matrix.topk(k=rank[-1], dim=-1, largest=False)[1]
    acc_list = []
    for r in rank:
        correct = (feature_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list

def train(model, optimizer):
    model.train() #model을 train용으로 설정
    
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels, _ in data_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        features, classes = model(inputs)
        
        class_loss = class_criterion(classes, labels)
        feature_loss = feature_criterion(features, labels)
        loss = class_loss + feature_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))
        
    return total_loss / total_num, total_correct / total_num * 100

def test(model, recall_ids):
    model.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels, _ in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                inputs, labels = inputs.to(device), labels.to(device)
                features, classes = model(inputs)
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        # compute recall metric
        acc_list = recall(eval_dict['test']['features'], test_dataset.labels, recall_ids)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
        writer.add_scalar("recall@{}/test".format(rank_id),float(results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)),epoch)
    print(desc)
    return acc_list[0]

if __name__ == '__main__':
    writer = SummaryWriter()
    
    batch_size = 50
    train_dataset = ClothImageDataset('train.csv',train=True)
    train_sample = MPerClassSampler(train_dataset.labels,batch_size,m=2)
    train_data_loader = DataLoader(train_dataset, batch_sampler=train_sample)

    test_dataset = ClothImageDataset('test.csv',train=False)
    test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    eval_dict = {'test': {'data_loader': test_data_loader}}

    backbone_type = 'resnet_50'
    gd_config = "MG"
    feature_dim = 512
    smoothing = 0.1
    temperature = 0.5
    margin = 0.1
    recalls =[1,2,4,8]
    num_epochs = 20
    num_classes = np.max([np.max(train_dataset.labels),np.max(test_dataset.labels)])+1
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format("clothes", backbone_type, gd_config, feature_dim, margin, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CGD(backbone_type, gd_config, feature_dim, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.8 * num_epochs)], gamma=0.1)
    class_criterion = nn.CrossEntropyLoss()
    feature_criterion = CustomTripletMarginLoss(margin=margin)

    best_recall = 0.0

    results = {}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []
        
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer)
        writer.add_scalar("Loss/train",train_loss,epoch)
        writer.add_scalar("Accuracy/train",train_accuracy,epoch)
        rank = test(model, recalls)
        lr_scheduler.step()
        
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['test_images'] = test_dataset.images
            data_base['test_labels'] = test_dataset.labels
            data_base['test_features'] = eval_dict['test']['features']
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))
    writer.flush()
    writer.close()