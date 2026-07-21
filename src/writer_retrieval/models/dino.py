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


class DINOModelv1(DINOModelBase):
    """
    Loads and constructs a DINOv1 model using TorchHub
    """
    
    def __init__(self, version: str, weights: str, device: torch.device):
        """
        Loads a DINOv1 model from TorchHub and sets up the weights, if available. NOTE: there are two
        ways that weights can be loaded:
         - Official DINO weight should be named "dino_<version>", otherwise this script assumes
         they are checkpoints
         - Checkpoint files gathered from training, this mode is used on any file that is not detected
         to be a dino weight file (files that are named "dino_<version>")
        
        Args:
            version (str): the version of the DINOv1 model to load. This should be among ["vits16", 
                "vits8", "vitb16", and "vitb8"]
            weights (str): the path to the weights that are loaded into the model. If `use_checkpoint`
                is `False`, then these should be official, pre-trained DINOv1 weights. If
                `use_checkpoint` is `True`, these weights act as a "checkpoint" loaded in with
                `load_state_dict`
        """
        
        if not weights.split('/')[-1].startswith("dino"):
            print("Loading DINOv1 from checkpoint file")
            self.model = torch.hub.load(
                "facebookresearch/dino:main",
                f"dino_{version}",
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
        
        self.model = self.model.to(device)
        self.model.eval()
    
    def extract_windows(self, windows: Tensor) -> Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.get_intermediate_layers(windows, n=1)[0]


class DINOModelv3(DINOModelBase):
    """
    Loads and constructs a DINOv3 model using TorchHub
    """
    
    def __init__(self, version: str, weights: str, device: torch.device):
        """
        Loads a DINOv3 model from TorchHub and sets up the weights, if available. NOTE: there are two
        ways that weights can be loaded:
         - Official DINO weight should be named "dinov3_<version>", otherwise this script assumes
         they are checkpoints
         - Checkpoint files gathered from training, this mode is used on any file that is not detected
         to be a dino weight file (files that are named "dinov3_<version>")
        
        Args:
            version (str): the version of the DINOv3 model to load. This should be among ["vits16",
            "vits16plus", "vitb16", "vitl16", "vith16plus", and "vit7b16"]
            weights (str): the path to the weights that are loaded into the model. If `use_checkpoint`
                is `False`, then these should be official, pre-trained DINOv3 weights. If
                `use_checkpoint` is `True`, these weights act as a "checkpoint" loaded in with
                `load_state_dict`
        """
        
        if not weights.split('/')[-1].startswith("dinov3"):
            print("Loading DINOv3 from checkpoint file")
            self.model = torch.hub.load(
                "facebookresearch/dinov3:main",
                f"dinov3_{version}",
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
        
        self.model = self.model.to(device)
        self.model.eval()
    
    def extract_windows(self, windows: Tensor) -> Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.forward_features(windows)['x_norm_patchtokens']

