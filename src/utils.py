import torch
import cv2
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def add_g(image_array, mean=0.0, var=30):
    """
    Add Gaussian noise to an image array.

    Parameters
    ----------
    image_array : np.ndarray
        Input image array.
    mean : float, optional
        Mean of the Gaussian noise. Defaults to 0.0.
    var : float, optional
        Variance of the Gaussian noise. Defaults to 30.

    Returns
    -------
    np.ndarray
        Noisy image array.
    """
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid



def adapt_first_layer(model):
    first_layer = model.conv1  # Assuming the first layer is named `conv1`
    new_layer = nn.Conv2d(1, first_layer.out_channels, kernel_size=first_layer.kernel_size, 
                          stride=first_layer.stride, padding=first_layer.padding, bias=False)
    new_layer.weight.data = first_layer.weight.data.mean(dim=1, keepdim=True)  # Average over RGB channels
    model.conv1 = new_layer
    return model

# Visualization function
def visualize_attention(image, attention_maps, class_idx=None):
    """
    Visualize attention maps over an image.
    
    Args:
        image (Tensor): Input image tensor (C, H, W).
        attention_maps (Tensor): Heatmaps from the model (N, num_classes, H, W).
        class_idx (int, optional): Specific class index to visualize. If None, visualizes all classes.
    """
    # Convert image to numpy
    img = ToPILImage()(image.cpu()).convert("RGB")
    img = np.array(img)

    # Normalize the attention maps
    attention_maps = attention_maps.cpu().detach().numpy()
    attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)

    if class_idx is not None:
        attention_maps = attention_maps[:, class_idx:class_idx+1, :, :]  # Only keep the selected class

    # Plot heatmaps
    num_classes = attention_maps.shape[1]
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    if num_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        heatmap = attention_maps[0, i]
        ax.imshow(img, alpha=0.6)
        ax.imshow(heatmap, cmap='jet', alpha=0.4)
        ax.axis('off')
        ax.set_title(f"Class {i}" if class_idx is None else f"Class {class_idx}")
    plt.show()
