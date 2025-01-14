# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import pickle
import wandb
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import RafDataset
from model import Model
from utils import *
from resnet import *
from loss import ACLoss

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='../raf-basic', help='raf_dataset_path')
parser.add_argument('--resnet50_path', type=str, default='../model/resnet50_ft_weight.pkl', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--ratio', '-r', type=float, default=0.3, help='ratio of label shuffle in training only 0,0.1,0.2,0.3')
args = parser.parse_args()


def train(args, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.to(device)
    model.train()

    total_loss = []
    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        labels = labels.to(device)


        criterion = nn.CrossEntropyLoss(reduction='none')



        output, hm1 = model(imgs1)
        output_flip, hm2 = model(imgs2)
        
        grid_l = generate_flip_grid(args.w, args.h, device)
        

        loss1 = nn.CrossEntropyLoss()(output, labels)
        flip_loss_l = ACLoss(hm1, hm2, grid_l, output)


        loss = loss1 + args.lam * flip_loss_l


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    return acc, running_loss


    
def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0


        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)


            outputs, _ = model(imgs1)


            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss
            data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
    return test_acc, running_loss
        
        
        
def main():    
    setup_seed(0)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    

    train_dataset = RafDataset(args, ratio=args.ratio, phase='train', transform=train_transforms)
    test_dataset = RafDataset(args, phase='test', transform=eval_transforms)
    print(f'train: {train_dataset.__len__()}, test: {test_dataset.__len__()}')    


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    
    
    
    model = Model(args)
    
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    run = wandb.init(project='raf-db', name='rebuttal_50_noise_'+str(args.label_path),
                     config={'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': 0.0001, 'weight_decay': 1e-4, 'ratio': args.ratio})
    
   
    
    
    for i in range(1, args.epochs + 1):
        train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
        test_acc, test_loss = test(model, test_loader, device)
        wandb.log({'Epoch': i, 'Train Loss': train_loss, 'Train Acc': train_acc, 'Test Loss': test_loss, 'Test Acc': test_acc})
        with open('rebuttal_50_noise_'+str(args.label_path)+'.txt', 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'\n')
    
    test_labels = []
    test_preds = []
    with torch.no_grad():
        model.eval()
        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)
            output, _ = model(imgs1)
            _, predicts = torch.max(output, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicts.cpu().numpy())
    test_cm = confusion_matrix(test_labels, test_preds)
    #save model 
    torch.save(model.state_dict(), 'rebuttal_50_noise_'+str(args.label_path)+'.pth')
    class_names = ['1', '2', '3', '4', '5', '6', '7']# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_cm, annot=True, cmap='Blues')

    # Labels, title and custom x-axis labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    plt.savefig('conf_mat.png', dpi=300)
    wandb.log({"Confusion Matrix": [wandb.Image("conf_mat.png", caption="Confusion Matrix")]})
    run.finish()





        
        



if __name__ == '__main__':
    main()
