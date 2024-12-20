# -*- coding: utf-8 -*-
import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
from utils import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase,ratio=0.0, basic_aug=True, transform=None):
        """
        Args:
            args: Arguments containing dataset configuration.
            phase: 'train' or 'test'. Determines which dataset split to load.
            basic_aug: Whether to apply basic augmentations.
            transform: Transformations to be applied on the images.
        """
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.ratio = ratio
        # Define image directories
        self.image_dir = os.path.join(self.raf_path, phase)

        # Collect all file paths and labels
        self.file_paths = []
        self.labels = []
        for label in os.listdir(self.image_dir):
            label_dir = os.path.join(self.image_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('_aligned.jpg'):
                        self.file_paths.append(os.path.join(label_dir, file_name))
                        self.labels.append(int(label) - 1)  # Labels are 1-based in the folder structure

        # Augmentation functions
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.random_erasing = transforms.RandomErasing(p=1, scale=(0.02, 0.2))
        self.aug_func = [flip_image, add_g]
        # self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        if self.phase == 'train':
            self.labelshuffle()
    #todo Randomly with ratio and seed change old lable to one of the other
    def labelshuffle(self):
        # random.seed(seed)
        num_labels = len(set(self.labels))
        for i in range(len(self.labels)):
            if random.uniform(0,1) < self.ratio:
                new_label = random.choice(list(set(range(num_labels)) - {self.labels[i]}))
                self.labels[i] = new_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(self.file_paths[idx])
            
        image = image[:, :, ::-1]
        
        
        if not self.clean:    
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.clean:
            image1 = self.random_erasing(image)

        return image, label, idx, image1

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    args = lambda: None
    args.raf_path = '../raf-basic'
    args.label_path = 'list_patition_label.txt'
    args.w = 7
    args.h = 7
    args.gpu = 0
    args.lam = 5
    args.epochs = 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = RafDataset(args, 'train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (image, label, idx, _) in enumerate(dataloader):
        print(image.shape, label.shape, idx.shape)
