# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
from utils import *


# A fer2013 dataset class from imagefolder that allow basic augmentations and shuffle labels

class FER2013Dataset(data.Dataset):
    def __init__(self, root_dir, phase = 'train', ratio=0.0, transform=None, shuffle_labels=False):
        """
        Args:
            root_dir (str): Path to the root directory containing image folders.
            transform (callable, optional): A function/transform to apply to the images.
            shuffle_labels (bool): Whether to shuffle labels for noisy label experiments.
        """
        self.root_dir = root_dir
        self.basic_aug = True
        self.transform = transform
        self.shuffle_labels = shuffle_labels
        self.ratio = ratio
        self.data = []
        self.labels = []

        # Load images and labels
        self.classes = os.listdir(root_dir)
        self.classes.sort()  # Ensure consistent class order
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            print(class_path)
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

        # Shuffle labels if required
        if self.shuffle_labels:
            self.labelshuffle()
        self.clean = True
        self.phase = phase
        # Augmentation functions
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.random_erasing = transforms.RandomErasing(p=1, scale=(0.02, 0.2))
        self.aug_func = [flip_image,add_g]

    def __len__(self):
        return len(self.data)
    
    def labelshuffle(self):
        # random.seed(seed)
        num_labels = len(set(self.labels))
        for i in range(len(self.labels)):
            if random.uniform(0,1) < self.ratio:
                new_label = random.choice(list(set(range(num_labels)) - {self.labels[i]}))
                self.labels[i] = new_label

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)
        # Apply transformations
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform:
            image = self.transform(image)

        if self.clean:
            image1 = self.flip_transform(image)

        return image, label, idx, image1

# Example usage
if __name__ == "__main__":
    # Define basic augmentations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create dataset instance
    dataset = FER2013Dataset(root_dir="/mnt/c/Freelancing/Grenwich/FGW/efficient-fer/Erasing-Attention-Consistency/data", transform=transform, shuffle_labels=True)

    # Check if first image is flipped
    image, label, idx, image1 = dataset[0]
    print(image.shape, label, idx, image1.shape)
    # Visualize
    import matplotlib.pyplot as plt
    