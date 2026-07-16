from writer_retrieval.data import WINDOW_SIZE

from torch import Tensor
from torch.nn.functional import pad

def pad_document(document: Tensor, stride: int = 224) -> Tensor:
    """
    Pads the `document` to a multiple of `stride`. This creates a new copy of the tensor
    """

    # get size
    h, w = document.size()[-2:]

    pad_w = (-w) % stride
    pad_h = (-h) % stride

    return pad(document, [0, pad_w, 0, pad_h])

def split_document(document: Tensor, stride: int = 224, window_size: int = WINDOW_SIZE) -> Tensor:
    """
    Returns a view of document` that's been split into a collection of windows of
    `window_size`x`window_size`
    """
    
    # This does a few things all at once:
    #  - Creates a new view using unfold, which creates additional dimensions to show where data is located
    #    this creates a tensor of shape: [channels, rows, cols, height, width]
    #  - Since we want the shape to be [windows, channels, height, width], we need to rearrange the dimensions
    #    and combine the rows and columns (permutate, then reshape)
    
    return document\
        .unfold(1, window_size, stride)\
        .unfold(2, window_size, stride)\
        .permute(1, 2, 0, 3, 4)\
        .reshape(-1, document.shape[0], window_size, window_size)