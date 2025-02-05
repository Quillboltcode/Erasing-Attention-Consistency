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
    # plt.show()

    # Save the plot
    plt.savefig(f'../log/attention_maps/attention_maps_{image_name}_{class_idx}.png')
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