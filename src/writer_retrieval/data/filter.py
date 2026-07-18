import torch
from torch import Tensor

def get_window_filter(windows: Tensor, threshold: float = 0.025) -> Tensor:
    """
    Returns a mask that can be used to filter out the set of `windows` that contain
    less than `threshold` percentage of foreground pixels
    
    Args:
        windows (Tensor): the windows to filter
        threshold (float): the proportion of foreground pixels each window much have
            in order to be
    
    Returns:
        window_mask (Tensor): the mask to be applied to the window tensors
    """
    
    return windows.mean(dim=(1, 2, 3)) >= threshold