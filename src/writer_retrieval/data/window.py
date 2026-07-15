from torch import Tensor
from torch.nn.functional import pad

def pad_document(document: Tensor, stride: int) -> Tensor:
    h, w = document.shape[-2:]
    
    pad_w = (-w) % stride
    pad_h = (-h) % stride
    
    # don't pad if not necessary
    if pad_w == 0 and pad_h == 0:
        return document
    
    # return padded document
    return pad(document, [0, pad_w, 0, pad_h])