from writer_retrieval.data.dataset import HistoricalWIDataset, WindowSampler, window_collate
from writer_retrieval.data.window import pad_document

from torch.utils.data import DataLoader

from tqdm import tqdm

if __name__ == "__main__":
    dataset = HistoricalWIDataset("datasets/historical_wi/train")
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=WindowSampler(dataset, dataset.total_windows, 512),
        collate_fn=window_collate
    )
    
    for batch in dataloader:
        print(len(batch))