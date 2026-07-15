from writer_retrieval.data.dataset import HistoricalWIDataset

if __name__ == "__main__":
    dataset = HistoricalWIDataset("historical_wi/test")
    
    print(dataset[5])