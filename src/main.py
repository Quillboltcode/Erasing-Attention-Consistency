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
from torchvision import transforms,datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy import percentile

from dataset import RafDataset
from fer2013 import FER2013Dataset
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
    for batch_i, (imgs1, labels, indexes, imgs2,_ ) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        labels = labels.to(device)


        criterion = nn.CrossEntropyLoss(reduction='none')



        output, hm1 = model(imgs1)
        output_flip, hm2 = model(imgs2)
        
        grid_l = generate_flip_grid(args.w, args.h, device)
        

        loss1 = nn.CrossEntropyLoss()(output, labels)
        wandb.log(("ce_loss", loss1))
        flip_loss_l = ACLoss(hm1, hm2, grid_l, output)
        wandb.log(("flip_loss", flip_loss_l))

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
    return acc, running_loss, loss1, flip_loss_l


    
def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0


        for batch_i, (imgs1, labels, indexes, imgs2, _) in enumerate(test_loader):
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
        
def find_high_flip_loss_images(args, model, data_loader, device):
    """
    Identify images with flip_loss_l higher than the mean loss of their class.
    
    Args:
        args: Arguments containing hyperparameters and settings.
        model: Trained model.
        data_loader: Data loader for the dataset.
        device: Device for computation (e.g., 'cuda' or 'cpu').
        
    Returns:
        high_loss_images (dict): Dictionary mapping class labels to lists of high-loss images.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Store losses and images grouped by class
    class_losses = defaultdict(list)
    class_images = defaultdict(list)
    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for imgs1, labels, _, imgs2, img_paths in data_loader:  # Ensure img_paths is used
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            labels = labels.to(device)

            # Forward pass for original and flipped images
            output, hm1 = model(imgs1)
            output_flip, hm2 = model(imgs2)

            # Generate flip grid
            grid_l = generate_flip_grid(args.w, args.h, device)

            # Calculate loss for each image in the batch
            flip_loss_l = []
            for i in range(len(labels)):
                hm1_single = hm1[i].unsqueeze(0)  # (1, 7, 7, 7)
                hm2_single = hm2[i].unsqueeze(0)  # (1, 7, 7, 7)
                loss = ACLoss(hm1_single, hm2_single, grid_l, output[i].unsqueeze(0))  # scalar
                flip_loss_l.append(loss.item())  # Convert scalar tensor to Python float

            # Collect losses, images, and paths by class
            for i in range(len(labels)):
                label = labels[i].item()
                loss = flip_loss_l[i]  # Already a Python float
                class_losses[label].append(loss)
                class_images[label].append({
                    'image': imgs1[i].cpu(),
                    'loss': loss,
                    'path': img_paths[i]  # Include image path
                })

    # Identify high-loss images and calculate mean losses
    # Calculate high-loss images using the IQR method
    high_loss_images = defaultdict(list)
    mean_losses = {}

    for label, losses in class_losses.items():
        # Calculate mean loss for reference
        mean_loss = sum(losses) / len(losses)
        mean_losses[label] = mean_loss  # Store the mean loss for each class

        # Calculate 75th percentile (Q3) and interquartile range (IQR)
        q3 = percentile(losses, 75)  # 75th percentile
        q1 = percentile(losses, 25)  # 25th percentile
        iqr = q3 - q1

        # Define threshold as Q3 + 1.5 * IQR
        threshold = q3 + 1.5 * iqr

        # Identify high-loss images based on the threshold
        for img_data in class_images[label]:
            if img_data['loss'] > threshold:
                high_loss_images[label].append(img_data)

    # Return both high-loss images and mean losses
    return {
        "high_loss_images": high_loss_images,
        "mean_losses": mean_losses
    }



        
def main():    
    setup_seed(0)
    
    # train_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    # eval_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])])
    trial_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    trial_dataset = datasets.ImageFolder(root=args.raf_path,transform=trial_transform)
    mean, std = calculate_mean_std(trial_dataset)
    print(f"Mean: {mean}, Std: {std}")
    # train_dataset = RafDataset(args, ratio=args.ratio, phase='train', transform=train_transforms)
    # test_dataset = RafDataset(args, phase='test', transform=eval_transforms)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean],
                             std=[std]),])
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean],
                             std=[std])])

    train_dataset = FER2013Dataset(root_dir=args.raf_path,phase='train',transform=train_transforms)
    val_dataset = FER2013Dataset(root_dir=args.raf_path, phase='val', transform=eval_transforms)
    test_dataset = FER2013Dataset(root_dir=args.raf_path, phase='test', transform=eval_transforms)
    print(f'train: {train_dataset.__len__()}, test: {test_dataset.__len__()}')    


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    
    
    
    model = Model(args)
    model = adapt_first_layer(model)    
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    run = wandb.init(project='fer2013-eac', name='rebuttal_50_noise_'+str(args.label_path),
                     config={'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': 0.0001, 'weight_decay': 1e-4, 'ratio': args.ratio})
    
   
    
    
    for i in range(1, args.epochs + 1):
        train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
        val_acc, val_loss = test(model, val_loader, device)
        wandb.log({'Epoch': i, 'Train Loss': train_loss, 'Train Acc': train_acc, 'val Loss': val_loss, 'val Acc': val_acc})
        with open('rebuttal_50_noise_'+str(args.label_path)+'.txt', 'a') as f:
            f.write(str(i)+'_'+str(val_acc)+'\n')

    torch.save(model.state_dict(), 'res_50_13_'+str(args.label_path)+'.pth')
    test_acc, test_loss = test(model, test_loader, device)
    wandb.log({'test Loss': test_loss, 'test Acc': test_acc})        
    test_labels = []
    test_preds = []
    with torch.no_grad():
        model.eval()
        for batch_i, (imgs1, labels, indexes, imgs2,_) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)
            output, _ = model(imgs1)
            _, predicts = torch.max(output, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicts.cpu().numpy())
    test_cm = confusion_matrix(test_labels, test_preds)
    test_cm = test_cm.astype('int')
    #save model 
    # fer2013 class name is label
    class_names= ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    #class_names = ['1', '2', '3', '4', '5', '6', '7']# 1:Surprise, 2:Fear, 3:Disgust, 4:Happiness, 5:Sadness, 6:Anger, 7:Neutral
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

    # todo: run on both train and val and test
    results = find_high_flip_loss_images(args, model, train_loader, device)

    # Extract high-loss images and mean losses
    high_loss_images = results["high_loss_images"]
    mean_losses = results["mean_losses"]

    # Print results for high-loss images
    # log with write to text file

    high_loss_table = results["high_loss_images"]
    with open('rebuttal_50_noise_'+str(args.label_path)+'.txt', 'a') as f:
        for cls, images in high_loss_images.items():
            f.write(f"Class {cls}: {len(images)} images with high flip loss\n")
            print(f"Class {cls}: {len(images)} images with high flip loss")
            for img_data in images:
                f.write(f"Loss: {img_data['loss']:.4f}")
                print(f"Loss: {img_data['loss']:.4f}")
                f.write(f"\tPath: {img_data['path']}\n")
                print(f"\tPath: {img_data['path']}")    
        # Print mean losses for each class
        for cls, mean_loss in mean_losses.items():
            print(f"Mean loss for class {cls}: {mean_loss:.4f}")
            f.write(f"Mean loss for class {cls}: {mean_loss:.4f}\n")
        
    

    run.finish()





        
        



if __name__ == '__main__':
    main()
