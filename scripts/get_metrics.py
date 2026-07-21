from pathlib import Path
from argparse import ArgumentParser

from writer_retrieval.data.serialization import load_patch
import writer_retrieval.models.vlad as vlad
import writer_retrieval.models.pca as pca

from writer_retrieval.retrieval.index import WriterIndex
import writer_retrieval.retrieval.metrics as metrics

import torch
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm

def retrieve_writers(root: str, name: str) -> None:
    """
    Loads all the testing data in `root/test` and passes it into a FAISS index for top-k and mAP metrics
    """
    
    paths = sorted(Path(root).joinpath("test").rglob("*"))
    descriptors: list[Tensor] = []
    writers: list[int] = []
    
    # load models
    codebook = vlad.VLADCodebook()
    codebook.load(f"output/models/{name}/vlad.pt")
    
    pca_model = pca.PCAMatrix()
    pca_model.load(f"output/models/{name}/pca.model")
    
    # load documents in batches
    for path in tqdm(paths, desc="Creating VLAD Descriptors"):
        for document, writer, doc_id in load_patch(path):
            # create VLAD descriptor
            descriptor = codebook.create_descriptor(document)
            
            # apply PCA whitening
            descriptor = torch.as_tensor(pca_model.apply(descriptor.unsqueeze(0)))
            descriptor = F.normalize(descriptor, p=2, dim=1)
            
            # add to list
            descriptors.append(descriptor)
            writers.append(writer)
    
    # collect descriptors
    descriptors: Tensor = torch.cat(descriptors)
    writers: Tensor = torch.as_tensor(writers)
    
    # create index
    index = WriterIndex(descriptors.shape[-1])
    index.add(descriptors)
    
    # calculate metrics
    met = metrics.Metrics(descriptors, writers, index)
    
    # calculate metrics
    met.run_metrics(
        "output/metrics/results.csv",
        name,
        ["topk:1", "topk:5", "topk:10", "mAP:-1", "mAP:5", "mAP:10"],
    )

if __name__ == "__main__":
    parser = ArgumentParser("Extract Patches")
    
    # add arguments
    parser.add_argument("root_path")
    parser.add_argument("run_name")
    
    # parse arguments
    args = parser.parse_args()
    
    retrieve_writers(args.root_path, args.run_name)