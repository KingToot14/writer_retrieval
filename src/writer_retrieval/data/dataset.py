from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import ToTensor

class HistoricalWIDataset(Dataset):
    """
    A custom dataset that loads and sets labels for HistoricalWI-style datasets. Specifically,
    writer labels are embedded into the file name as <writer_id>-<document_name>.<ext>
    """
    
    def __init__(self, root: str | Path, transform = ToTensor()):
        """
        Creates a new Dataset by recursively gathering all files in the given root directory.
        This only collects files that are readable by the Pillow library, so it should avoid
        metadata and other files
        """
        self.transform = transform
        
        # get accepted file types
        extensions: dict[str, str] = Image.registered_extensions()
        supported: set[str] = {ext for ext, f in extensions.items() if f in Image.OPEN}
        
        # fetch all file names
        self.samples: list[tuple[Path, int]] = []
        
        for path in sorted(Path(root).rglob("*")):
            # make sure file extension is supported
            if not path.suffix.lower() in supported:
                continue
            
            self.samples.append((path, path.name.split("-", 1)[0]))
    
    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, int]:
        """
        Loads the requested image file and determines its label from the file name
        """
        path, label = self.samples[index]
        
        # load file
        image: Image.Image = None
        
        with Image.open(path) as img:
            image = img.convert("L")
        
        # do transformation
        if self.transform != None:
            image = self.transform(image)
        
        return image, label