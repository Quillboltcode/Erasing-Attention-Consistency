from main import find_high_flip_loss_images
from fer2013 import FER2013Dataset
from model import Model
from utils import *
from resnet import *
from loss import ACLoss


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# Mock Dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, img_size, num_classes):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.file_paths = [f'path/to/image/{i}' for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random grayscale images (1 channel)
        img1 = torch.rand(1, self.img_size, self.img_size)
        img2 = torch.rand(1, self.img_size, self.img_size)  # Flipped version
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img1, label, idx, img2, self.file_paths[idx]

# Mock Model


# Test Function
def test_find_high_flip_loss_images():
    args = type('', (), {})()  # Mock arguments
    args.w, args.h = 7, 7  # Heatmap size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create random dataset and dataloader
    dataset = FER2013Dataset(root_dir="/mnt/c/Freelancing/Grenwich/FGW/efficient-fer/Erasing-Attention-Consistency/data", transform=None, shuffle_labels=True)
    data_loader = DataLoader(dataset, batch_size=15, shuffle=True)

    # Initialize mock model
    model = Model(args=None)
    model = adapt_first_layer(model)
    # Call the function
    results = find_high_flip_loss_images(args, model, data_loader, device)

    # Extract high-loss images and mean losses
    high_loss_images = results["high_loss_images"]
    mean_losses = results["mean_losses"]

    # Print results for high-loss images
    with open("high_loss_images.txt", "w") as f:
        for cls, images in high_loss_images.items():
            print(f"Class {cls}: {len(images)} images with high flip loss")
            f.write(f"Class {cls}: {len(images)} images with high flip loss\n")
            for img_data in images:
                print(f" - Loss: {img_data['loss']:.4f}")
                f.write(f" - Loss: {img_data['loss']:.4f}")
                print(f" - Path: {img_data['path']}")
                f.write(f"\t - Path: {img_data['path']}\n")

    # Print mean losses for each class
    for cls, mean_loss in mean_losses.items():
        print(f"Mean loss for class {cls}: {mean_loss:.4f}")
# Run the test
test_find_high_flip_loss_images()
