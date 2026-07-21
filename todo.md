# Overview
I'm going to be basically re-implementing everything since the way I was doing it before was poorly structured and mixed way to many different aspects of the pipeline into weird scripts. This time, I'm going to go for a much more modular design, with actual training scripts in order to run the entire pipeline efficiently

## Components
1. Feature Extraction
   - Instead of just using the raw patch features, I'm going to perform a few pre-processing steps
   - First, we need to calculate the percentage of foreground pixels in each given training window
   - Then, we need to calculate the number of foreground pixels in each individual patch (14x14), discarding the patches below a certain threshold
   - This will be implemented purely in PyTorch, exported to a save file for another script
   - This saves each document as a single .pt file for later use (or broken-up runs)
2. VLAD Codebook
   - Once we have all the extracted foreground features, we can build the VLAD codebook from the training documents
   - This will use faiss as it's fast and compatible with PyTorch tensors, specifically their kmeans clustering method
   - Apply power normalization and l2 normalization
   - Fit a PCA whitening model (also faiss) using the encoded training descriptors
   - Store the original centroids as a .np file, and the PCA using faiss's export methods
3. Generate Descriptors
   - Load each document one-by-one and compute the VLAD descriptor
   - Process: find the closest clusters to each descriptor, sum the residuals, flatten the final array
   - For each of these, the results are stored as a .np array
4. Compare Documents
   - Load each aggregated document into a faiss index
   - Using this (implementation to come), determine top-k accuracy (1, 5, 10), mAP (all, @5, @10), recall (@5, @10, @25)
   - Store results in a .csv file (append)

## Flow
The basic flow will be structured as follows:
- Create a Dataset and Dataloader for the training set
- In batches, split the documents into windows, skipping windows with a low foreground percentage
   - First, pad documents to a multiple of `stride` using PyTorch's `Pad` transform
- In the same batches, extract the patch features
- During filtering, examine each document and return a list of patches to keep (foreground threshold), and apply it
   - This can be done using `avg_pool2d * 16 * 16`, comparing each kernel to `foreground_threshold`
- Take all training documents and build a VLAD codebook using a max of `100,000` or `1,000,000` random descriptors
   - This is done using faiss's `Kmeans` method for fast GPU training
   - Apply power normalization
   - Apply l2 normalization
   - Train a `PCAMatrix` (also faiss) using the final training descriptors (from VLAD)
- Free the other Dataset (if necessary), and create a new one for the testing set
- Run each testing document through VLAD
   - Find the nearest cluster to each descriptor
   - Sum the residuals per-cluster
   - Run through the pre-trained PCA model
- Load each aggregated document into a faiss index and run metrics
   - Calculate top-k accuracy (1, 5, 10), mAP (all, 5, 10), and recall (5, 10, 25)
   - Store results in a csv file

Note: this can be done either all at once (likely better for performance and clarity) or broken up into smaller steps

## Notes:
Extraction times (HistoricalWI test: 3,600 images):
- no_grad:              2:44
- inference_mode:       2:46
- float16:              1:26
- bfloat16:             1:30
- persist & prefetch:   1:26 (Presistent workers hangs indefinitely after completion, so it's not recommended to be used)
- workers: 16 => 0:     1:17
- workers=4:            1:22 (Starting with this one, prefetch_factor was set to 4)
- workers=8:            1:24
- workers=2:            1:19
- workers=0, compiled:  1:17
- window_count=8192:    1:19 (now possible due to half-precision)
- __getitems__:         1:16 (not really meaningfully different, but cool to have)
- filtered:             1:14

Updated benchmarking (full pipeline implemented):
- baseline:             0:34, 2:32
- tensorfloat32:        0:33, 2:36
- matmul_precision:     0:34, 2:37  (all pretty much insignificant)
- two gpus:             0:25, 1:33
- windows=8192:         0:27, 1:36
- windows=2048:         0:25, 1:36