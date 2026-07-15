from writer_retrieval.data.dataset import HistoricalWIDataset
from writer_retrieval.data.window import pad_document

from torch.utils.data import DataLoader

from tqdm import tqdm

if __name__ == "__main__":
    dataset = HistoricalWIDataset("datasets/historical_wi/train")
    
    print(dataset[5])