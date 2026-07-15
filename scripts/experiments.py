from PIL import Image
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import ToTensor

from time import time

from tqdm import tqdm

if __name__ == "__main__":
    start = time()
    for i in tqdm(range(10_000)):
        image = decode_image("datasets/historical_wi/train/7-IMG_MAX_10038.png", ImageReadMode.GRAY)
    print(time() - start)
    
    start = time()
    trans = ToTensor()
    for i in tqdm(range(10_000)):
        with Image.open("datasets/historical_wi/train/7-IMG_MAX_10038.png") as img:
            trans(img.convert("L"))
    print(time() - start)
