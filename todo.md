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
5. Reranking?
   - Honestly not too sure what this is yet, or how to implement it, but I'm willing to try it if it improve performance