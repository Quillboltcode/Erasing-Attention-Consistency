import torch.nn.functional as F
from torch.autograd import Variable

def ACLoss(att_map1, att_map2, grid_l, output):
    """
    Compute Attention Consistency Loss (AC Loss) between two attention maps.

    Parameters
    ----------
    att_map1 : torch.Tensor
        The first attention map.
    att_map2 : torch.Tensor
        The second attention map.
    grid_l : torch.Tensor
        The grid for flipping the second attention map.
    output : torch.Tensor
        The output of the model.

    Returns
    -------
    flip_loss_l : torch.Tensor
        The attention consistency loss between the two attention maps.
    """
    # Expand the flipping grid to match the batch size of the output and convert it to a variable
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad=False)
    
    # Rearrange the dimensions of the flipping grid to be compatible with grid_sample
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    
    # Apply the flipping grid to the second attention map to create a flipped version
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode='bilinear', padding_mode='border', align_corners=True)
    
    # Calculate the mean squared error loss between the original and flipped attention maps
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip)
    
    # Return the computed attention consistency loss
    return flip_loss_l
    