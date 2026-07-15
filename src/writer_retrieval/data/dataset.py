from math import ceil
from pathlib import Path

from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision.io import decode_image, ImageReadMode

# --- Constants --- #
EXTENSIONS = [".png", ".jpg", ".jpeg"]
WINDOW_SIZE = 224

# --- Classes --- #
class HistoricalWIDataset(Dataset):
    """
    A custom dataset that loads and sets labels for HistoricalWI-style datasets. Specifically,
    writer labels are embedded into the file name as <writer_id>-<document_name>.<ext>
    """
    
    @property
    def total_windows(self):
        """The total amount of windows this dataset contains based on the target stride"""
        return self._total_windows
    
    def __init__(self, root: str | Path, stride: int = WINDOW_SIZE):
        """
        Creates a new Dataset by recursively gathering all files in the given root directory.
        This only collects PNG and JPEG files, so it should avoid any metadata or non-image files
        """
        
        self._total_windows = 0
        
        # fetch all file names
        self.samples: list[tuple[Path, int]] = []
        
        for path in sorted(Path(root).rglob("*")):
            # make sure file extension is supported
            if path.suffix.lower() not in EXTENSIONS:
                continue
            
            # collect image size (shouldn't image into memory)
            w, h = 0, 0
            with Image.open(path) as img:
                w, h = img.size
            
            # calculate window count
            windows = (ceil((w - WINDOW_SIZE) / stride) + 1) * (ceil((h - WINDOW_SIZE) / stride) + 1)
            self._total_windows += windows
            
            self.samples.append((path, path.name.split("-", 1)[0], windows))
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        path, label, windows = self.samples[index]
        
        return decode_image(path, ImageReadMode.GRAY), label, windows

class WindowSampler(Sampler):
    """
    A special sampler that uses a window limit instead of a batch size. This limit is soft, so it
    stops adding windows after this limit has been reached, but almost always will exceed this limit
    """
    
    def __init__(self, data: HistoricalWIDataset, total_windows: int, max_windows: int):
        self.data: HistoricalWIDataset = data
        self.total_windows: int = total_windows
        self.max_windows: int = max_windows
    
    def __iter__(self):
        batch = []
        batch_wins = 0
        
        for i, item in enumerate(self.data):
            batch.append(i)
            
            # update total windows
            _, _, windows = item
            batch_wins += windows
            
            # yield the full batch
            if batch_wins >= self.max_windows:
                yield batch
                
                batch = []
                batch_wins = 0
        
        # yield last batch
        if batch:
            yield batch
    
    def __len__(self) -> int:
        return self.total_windows

def window_collate(data: list[tuple]):
    return data