import torch
from torch import Tensor

class DINOModelBase:
    """
    A base class that should not be instantiated directly. This exposes a single API for
    supporting multiple different DINO models and weights
    """
    
    def extract_windows(self, windows: Tensor) -> Tensor:
        """
        Extracts the patch features from the given tensor of `windows`. This should be in the
        shape: [batch, channels, height, width] where `height` and `width` are both `224`
        """
        pass
    
    def to(self, device: torch.device):
        """
        Wrapper around `self.model` to move this model to a specific `device`
        """
        self.model = self.model.to(device)
        
        return self

class DINOModelv1(DINOModelBase):
    """
    Loads and constructs a DINOv1 model using TorchHub
    """
    
    def __init__(self, version: str, weights: str, use_checkpoint: bool = True):
        """
        Loads a DINOv1 model from TorchHub and sets up the weights, if available.
        
        Args:
            version (str): the version of the DINOv1 model to load. This should be among ["dino_vits16", 
                "dino_vits8", "dino_vitb16", and "dino_vitb8"]
            weights (str): the path to the weights that are loaded into the model. If `use_checkpoint`
                is `False`, then these should be official, pre-trained DINOv1 weights. If
                `use_checkpoint` is `True`, these weights act as a "checkpoint" loaded in with
                `load_state_dict`
            use_checkpoint (bool): If `True`, this signals that the `weights` are using a checkpoint format
                rather than the official DINOv1 weight format
        """
        
        if use_checkpoint:
            print("Loading DINOv1 from checkpoint file")
            self.model = torch.hub.load(
                "facebookresearch/dino:main",
                version,
                pretrained=False
            )
            
            # load checkpoint
            checkpoint = torch.load(weights)
        
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "teacher" in checkpoint:
                checkpoint = checkpoint["teacher"]

            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            
            print("Missing:")
            print(missing)
            print("Unused:")
            print(unexpected)
        else:
            print("Loading DINOv1 from pre-trained weights")
            self.model = torch.hub.load(
                "facebookresearch/dino:main",
                version,
                weights=weights
            )
        
        self.model = self.model.cuda()
        self.model.eval()
    
    def extract_windows(self, windows: Tensor) -> Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.get_intermediate_layers(windows, n=1)[0]

class DINOModelv3(DINOModelBase):
    """
    Loads and constructs a DINOv3 model using TorchHub
    """
    
    def __init__(self, version: str, weights: str, use_checkpoint: bool = True):
        """
        Loads a DINOv3 model from TorchHub and sets up the weights, if available.
        
        Args:
            version (str): the version of the DINOv3 model to load. This should be among ["dinov3_vits16",
            "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", and "dinov3_vit7b16"]
            weights (str): the path to the weights that are loaded into the model. If `use_checkpoint`
                is `False`, then these should be official, pre-trained DINOv3 weights. If
                `use_checkpoint` is `True`, these weights act as a "checkpoint" loaded in with
                `load_state_dict`
            use_checkpoint (bool): If `True`, this signals that the `weights` are using a checkpoint format
                rather than the official DINOv3 weight format
        """
        
        if use_checkpoint:
            print("Loading DINOv3 from checkpoint file")
            self.model = torch.hub.load(
                "facebookresearch/dinov3:main",
                version,
                pretrained=False
            )
            
            # load checkpoint
            checkpoint = torch.load(weights)
        
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "teacher" in checkpoint:
                checkpoint = checkpoint["teacher"]

            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            
            print("Missing:")
            print(missing)
            print("Unused:")
            print(unexpected)
        else:
            print("Loading DINOv3 from pre-trained weights")
            self.model = torch.hub.load(
                "facebookresearch/dinov3:main",
                version,
                weights=weights
            )
        
        self.model = self.model.cuda()
        self.model.eval()
    
    def extract_windows(self, windows: Tensor) -> Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.forward_features(windows)['x_norm_patchtokens']

