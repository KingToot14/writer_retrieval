from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import decode_image, ImageReadMode

# --- Constants --- #
EXTENSIONS = [".png", ".jpg", ".jpeg"]

# --- Classes --- #
class HistoricalWIDataset(Dataset):
    """
    A custom dataset that loads and sets labels for HistoricalWI-style datasets. Specifically,
    writer labels are embedded into the file name as <writer_id>-<document_name>.<ext>
    """
    
    def __init__(self, root: str | Path):
        """
        Creates a new Dataset by recursively gathering all files in the given root directory.
        This only collects PNG and JPEG files, so it should avoid any metadata or non-image files
        """
        
        # fetch all file names
        self.samples: list[tuple[Path, int]] = []
        
        for path in sorted(Path(root).rglob("*")):
            # make sure file extension is supported
            if path.suffix.lower() not in EXTENSIONS:
                continue
            
            self.samples.append((path, path.name.split("-", 1)[0]))
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.samples[index]
        
        image: Tensor = decode_image(path, ImageReadMode.GRAY)
        
        return image, label
