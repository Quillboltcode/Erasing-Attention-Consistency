import torch
import cv2
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision import datasets
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
# from scipy.stats import entropy 
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
    first_layer = model.features[0]  # Assuming the first layer is named `conv1`
    new_layer = nn.Conv2d(1, first_layer.out_channels, kernel_size=first_layer.kernel_size, 
                          stride=first_layer.stride, padding=first_layer.padding, bias=False)
    new_layer.weight.data = first_layer.weight.data.mean(dim=1, keepdim=True)  # Average over RGB channels
    model.features[0] = new_layer
    return model

def calculate_mean_std(dataset):
    """
    Calculate mean and standard deviation of a dataset.
    Args:
        dataset: PyTorch dataset.
    Returns:
        mean, std: Calculated mean and standard deviation.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        images = images.view(images.size(0), -1)  # Flatten the images
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        total_samples += images.size(0)

    mean /= total_samples
    std /= total_samples
    return mean.item(), std.item()

def visualize_attention(image, flip_image, attention_maps, flip_attention_maps, image_name="",class_idx=None):
    """
    Visualize attention maps for original and flipped images side by side with finer and smoother heatmaps.

    Args:
        image (Tensor): Input original image tensor (C, H, W).
        flip_image (Tensor): Input flipped image tensor (C, H, W).
        attention_maps (Tensor): Heatmaps from the model for the original image (N, num_classes, H, W).
        flip_attention_maps (Tensor): Heatmaps from the model for the flipped image (N, num_classes, H, W).
        class_idx (int, optional): Specific class index to visualize. If None, visualizes all classes.
    """
    import torch
    from torchvision.transforms import ToPILImage
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from torch.nn.functional import interpolate

    # Convert images to numpy
    img = ToPILImage()(image).convert("RGB")
    img = np.array(img)

    flip_img = ToPILImage()(flip_image).convert("RGB")
    flip_img = np.array(flip_img)

    # Normalize and smooth the attention maps
    attention_maps = attention_maps.cpu().detach()
    # measure attention metrics wwith entropy, gini and attention spread
    attention_entropy_metric = attention_entropy(attention_maps)
    attention_spread_metric = attention_spread(attention_maps)
    attention_gini = gini_coefficient(attention_maps) 
    print(f'Attention entropy: {attention_entropy_metric}')
    print(f'Attention spread: {attention_spread_metric}')
    print(f'Attention gini: {attention_gini}')
    
    attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)
    attention_maps = interpolate(attention_maps, size=(224, 224), mode='bilinear', align_corners=False).numpy()
    attention_maps = gaussian_filter(attention_maps, sigma=2)

    flip_attention_maps = flip_attention_maps.cpu().detach()
    flip_attention_maps = (flip_attention_maps - flip_attention_maps.min()) / (flip_attention_maps.max() - flip_attention_maps.min() + 1e-8)
    flip_attention_maps = interpolate(flip_attention_maps, size=(224, 224), mode='bilinear', align_corners=False).numpy()
    flip_attention_maps = gaussian_filter(flip_attention_maps, sigma=2)

    if class_idx is not None:
        attention_maps = attention_maps[:, class_idx:class_idx+1, :, :]
        flip_attention_maps = flip_attention_maps[:, class_idx:class_idx+1, :, :]

    # Plot attention maps for original and flipped images
    num_classes = attention_maps.shape[1]
    fig, axes = plt.subplots(2, num_classes, figsize=(15, 10))
    if num_classes == 1:
        axes = [[axes[0]], [axes[1]]]

    for i in range(num_classes):
        # Original image attention map
        heatmap = attention_maps[0, i]
        axes[0][i].imshow(img, alpha=0.9)
        axes[0][i].imshow(heatmap, cmap='jet', alpha=0.3)
        axes[0][i].axis('off')
        axes[0][i].set_title(f"Original - Class {i}" if class_idx is None else f"Original - Class {class_idx}")

        # Flipped image attention map
        flip_heatmap = flip_attention_maps[0, i]
        axes[1][i].imshow(flip_img, alpha=0.9)
        axes[1][i].imshow(flip_heatmap, cmap='jet', alpha=0.3)
        axes[1][i].axis('off')
        axes[1][i].set_title(f"Flipped - Class {i}" if class_idx is None else f"Flipped - Class {class_idx}")

    plt.tight_layout()
    plt.show()

    # # Save the plot
    # plt.savefig(f'../log/attention_maps_mediapipe/attention_maps_{image_name}_{class_idx}.png')
    plt.close()


def process_image(img_name, net, MEAN, STD, utils, class_idx=5):
    """
    Process an image to calculate and compare attention maps for original and flipped images.

    Args:
        img_name (str): Name of the image file.
        net (torch.nn.Module): Trained model to compute attention maps.
        MEAN (float): Mean for normalization.
        STD (float): Standard deviation for normalization.
        utils (module): Utility module with helper functions.
        class_idx (int): Class index for attention visualization.
    """
    import cv2
    import torch
    from torchvision import transforms

    # Define the image path
    img_path = f'../data/org_fer2013/train/{img_name}'

    # Define evaluation transforms
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD])
    ])

    # Read and preprocess the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_t = eval_transforms(img)
    img_t = img_t.unsqueeze(0)

    # Compute attention map for the original image
    with torch.no_grad():
        _, att = net(img_t)

    # Process the flipped image
    flip_img = utils.flip_image(img)
    flip_img_t = eval_transforms(flip_img)
    flip_img_t = flip_img_t.unsqueeze(0)

    # Compute attention map for the flipped image
    with torch.no_grad():
        _, flip_att = net(flip_img_t)
    new_img_name = img_name.replace("/", "_").replace(".jpg", "")
    # Visualize attention maps side by side
    visualize_attention(img, flip_img, att, flip_att, new_img_name,class_idx=class_idx)


# Shanon entropy to quantify focus

def attention_entropy(attn_map):
    """
    Compute the entropy of the attention map to measure focus vs. spread.
    
    Parameters:
        attn_map (torch.Tensor): Attention map of shape (1, C, H, W).
        
    Returns:
        float: Mean entropy across channels.
    """
    # Ensure it's on the correct device and detach from computation graph
    attn_map = attn_map.detach()

    # Flatten spatial dimensions (keep batch & channels)
    B, C, H, W = attn_map.shape
    attn_map = attn_map.view(B, C, -1)  # Shape: (1, C, H*W)

    # Normalize attention values to sum to 1 per channel
    attn_sum = attn_map.sum(dim=-1, keepdim=True)
    attn_sum[attn_sum == 0] = 1  # Avoid division by zero
    attn_map = attn_map / attn_sum  

    # Apply small epsilon to avoid log(0) issues
    attn_map = torch.clamp(attn_map, min=1e-8, max=1)  # Clamping ensures valid log input

    # Compute entropy: H = -sum(p * log(p))
    entropy = -torch.sum(attn_map * torch.log(attn_map), dim=-1)  # Shape: (1, C)

    # Return mean entropy across channels
    return entropy.mean().item()

def gini_coefficient(attn_map):
    """
    Compute the Gini coefficient for an attention map of shape (1, C, H, W).
    
    Parameters:
        attn_map (numpy array or torch.Tensor): Attention map of shape (1, C, H, W).
        
    Returns:
        float: Gini coefficient (0 = equal distribution, 1 = highly concentrated).
    """
    # Ensure it's a NumPy array
    if isinstance(attn_map, np.ndarray):
        attn_map = attn_map.flatten()  # Flatten to (C*H*W)
    else:
        attn_map = attn_map.detach().cpu().numpy().flatten()  # Convert PyTorch tensor to NumPy & flatten

    # Ensure non-negative values (important for stability)
    attn_map = np.clip(attn_map, 0, None)

    # Normalize to prevent negative Gini values
    attn_map = attn_map / (attn_map.sum() + 1e-8)  # Avoid division by zero

    # Sort values (ascending)
    attn_map.sort()

    # Compute Gini coefficient using the standard formula
    n = len(attn_map)
    index = np.arange(1, n + 1)  # Rank index
    gini = (2 * np.sum(index * attn_map) / np.sum(attn_map) - (n + 1)) / n

    return max(0.0, min(gini, 1.0))  # Clamp between 0 and 1 to ensure validity



def attention_spread(attn_map):
    """
    Compute the spread of the attention map in a PyTorch tensor.
    
    Parameters:
        attn_map (torch.Tensor): Attention map of shape (1, C, H, W).
        
    Returns:
        float: Mean attention spread across channels.
    # """
    # print("Shape of attn_map:", attn_map.shape)  # Debugging

    _, C, H, W = attn_map.shape  # Extract dimensions

    # Create meshgrid for pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=attn_map.device), 
                          torch.arange(W, device=attn_map.device), indexing='ij')

    # Convert to float for computations
    x, y = x.float(), y.float()

    spread_values = []
    for c in range(C):
        attn = attn_map[0, c]  # Extract attention for this channel
        
        attn_sum = attn.sum()
        if attn_sum == 0:  
            spread_values.append(0.0)  # Avoid division by zero
            continue

        # Compute weighted mean location
        x_mean = (x * attn).sum() / attn_sum
        y_mean = (y * attn).sum() / attn_sum

        # Compute Euclidean distance from center
        center_x, center_y = W // 2, H // 2
        spread = torch.sqrt((x_mean - center_x) ** 2 + (y_mean - center_y) ** 2)

        spread_values.append(spread.item())  # Convert tensor to float

    # Return mean spread across channels
    return sum(spread_values) / len(spread_values) if spread_values else 0.0
  