# In[]
import torch
import anndata
from pathlib import Path
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import gc

import os

def is_within_uint32_range(value):
    uint32_max = 4_294_967_295
    return value <= uint32_max

# In[]
# ----------------------------------------------------------------------------
#
# Extract & Preprocess the data
#
# ----------------------------------------------------------------------------
# # match anchor of gene names
# geneid = np.loadtxt("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/geneid_select.txt", dtype = object)
# output_dir = "/localscratch/ziqi/hs_download/ordered/"
# data_dir = "/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/"
# # grouping of the gene expression counts according to the meta-genes
# cum_idx = 0
# tissues = ["all_remain", "pancreas", "blood", "brain", "heart", "intestine", "kidney", "lung"]
# # tissues = ["pancreas"]
# sizes = []
# for tissue in tissues:
#     input_dir = Path(f"{data_dir}{tissue}/")
#     files = [f for f in input_dir.glob("*.h5ad") if f.name[:9] == "partition"]
#     counts_metas = []
#     counts_metas_bin = []
#     meta_cells = []
#     print(f"{tissue}: found {len(files)} files in {input_dir}")
#     for idx, file in enumerate(files):
#         print(f"process chunk {cum_idx}...")
#         try:
#             sample_adata = anndata.read_h5ad(file)
#         except:
#             print(f"chunk {idx} corrupted, skip...")
#             continue
#         sample_adata.var.index = sample_adata.var["feature_id"].values
#         # select only the related genes
#         sample_adata = sample_adata[:, geneid].copy()
        
#         obs = sample_adata.obs
#         var = sample_adata.var
#         print(sample_adata.shape)

#         # NOTE: skip normalization, leave it to the training script
#         X_norm = sample_adata.X

#         # normalize 
#         # print("normalization...")
#         # X = sample_adata.X.toarray()
#         # libsize = X.sum(axis = 1)
#         # X_norm = X/(libsize[:, None] + 1e-6) * 10e4
#         # X_norm = np.log1p(X_norm)
        
#         assert X_norm.shape[1] == 18790
        
#         # # transform X_norm to sparse for efficient saving
#         # X_norm = sp.csr_matrix(X_norm)



#         print("save results...")
        
#         obs.to_parquet(os.path.join(output_dir, f"obs_{cum_idx}.parquet"))
#         if not os.path.exists(os.path.join(output_dir, "var.csv")):
#             # same across partitions
#             var.to_csv(os.path.join(output_dir, "var.csv"))

#         # extract the data and save 
#         data = X_norm.data.astype(np.float32)
#         data_memmap = np.memmap(os.path.join(output_dir, f"counts_data_{cum_idx}.npz"), dtype='float32', mode='w+', shape=data.shape)
#         data_memmap[:] = data[:]
#         data_memmap.flush()
        
#         # extract the indices and save
#         indices = X_norm.indices.astype(np.int16)
#         indices_memmap = np.memmap(os.path.join(output_dir, f"counts_indices_{cum_idx}.npz"), dtype='int16', mode='w+', shape=indices.shape)
#         indices_memmap[:] = indices[:]
#         indices_memmap.flush()

#         # extract the indptr and save, the value can be extremely large
#         indptr = X_norm.indptr.astype(np.uint32)
#         indptr_memmap = np.memmap(os.path.join(output_dir, f"counts_indptr_{cum_idx}.npz"), dtype='uint32', mode='w+', shape=indptr.shape)
#         # make sure the value can be saved with int32
#         assert is_within_uint32_range(np.max(indptr))
#         indptr_memmap[:] = indptr[:]
#         indptr_memmap.flush()

#         sizes.append([data.shape[0], indices.shape[0], indptr.shape[0]])

#         cum_idx += 1

#         del sample_adata, X_norm, obs
#         gc.collect()

#         print("Done.")

# sizes = np.array(sizes)
# np.savetxt(os.path.join(output_dir, "sizes.txt"), sizes)

# In[]
# Merge the datasets across partitions
data_dir = "/localscratch/ziqi/hs_download/ordered/"
data_dir = "/data/zzhang834/hs_healthy_2025_01_30/ordered/"
sizes = np.loadtxt(data_dir + "sizes.txt")

# read meta-data
var = pd.read_csv(data_dir + "var.csv", index_col = 0)
ngenes = var.shape[0]

# for concatenation, calculate the merged size
sizes_new = [0, 0, 0]
for partition_idx in range(0, len(sizes)):

    data_size = sizes[partition_idx, 0]
    indices_size = sizes[partition_idx, 1]
    indptr_size = sizes[partition_idx, 2]
    if partition_idx > 0:
        indptr_size -= 1
    
    sizes_new[0] += data_size
    sizes_new[1] += indices_size
    sizes_new[2] += indptr_size

print("merge data")
data_cum = np.memmap(os.path.join(data_dir, f"counts_data_cum.npz"), dtype = "float32", mode = "w+", shape = (int(sizes_new[0]), ))
current_row = 0
for partition_idx in range(0, len(sizes)):
    print(partition_idx)
    data = np.memmap(os.path.join(data_dir, f"counts_data_{partition_idx}.npz"), dtype="float32", mode="r", shape=(int(sizes[partition_idx, 0]),))
    # Determine the number of rows in the input file
    num_rows = data.shape[0]
    # Copy the data from input file to the appropriate section of the output memmap
    data_cum[current_row:current_row + num_rows] = data[:]
    # Update the row pointer for the next file
    current_row += num_rows
    # remove the data file
    # os.remove(os.path.join(data_dir, f"counts_data_{partition_idx}.npz"))
# Flush changes to the output file
data_cum.flush()

print("merge indices")
indices_cum = np.memmap(os.path.join(data_dir, f"counts_indices_cum.npz"), dtype = "int16", mode = "w+", shape = (int(sizes_new[1]), ))
current_row = 0
for partition_idx in range(0, len(sizes)):
    print(partition_idx)
    indices = np.memmap(os.path.join(data_dir, f"counts_indices_{partition_idx}.npz"), dtype= "int16", mode= "r", shape=(int(sizes[partition_idx, 1]),))
    # Determine the number of rows in the input file
    num_rows = indices.shape[0]
    # Copy the data from input file to the appropriate section of the output memmap
    indices_cum[current_row:current_row + num_rows] = indices[:]
    # Update the row pointer for the next file
    current_row += num_rows
    # remove the data file
    # os.remove(os.path.join(data_dir, f"counts_indices_{partition_idx}.npz"))
# Flush changes to the output file
indices_cum.flush()

print("merge indptr")
# Create the output memmap file with the total size
indptr_cum = np.memmap(os.path.join(data_dir, f"counts_indptr_cum.npz"), dtype = "uint64", mode = "w+", shape = (int(sizes_new[2]), ))
# Copy data from each input file into the output memmap file
current_row = 0
for partition_idx in range(0, len(sizes)):
    print(partition_idx)
    # update the indptr for concatenation
    indptr = np.memmap(os.path.join(data_dir, f"counts_indptr_{partition_idx}.npz"), dtype = "uint32", mode = "r", shape=(int(sizes[partition_idx, 2]),))
    # adjust in an eariler stage
    indptr = indptr.astype(np.uint64)
    assert np.min(indptr) >= 0
    if partition_idx > 0:
        indptr = indptr[1:] + indptr_last
    indptr_last = indptr[-1]
    # Determine the number of rows in the input file
    num_rows = indptr.shape[0]
    # Copy the data from input file to the appropriate section of the output memmap
    indptr_cum[current_row:current_row + num_rows] = indptr[:]
    # Update the row pointer for the next file
    current_row += num_rows
    # remove the data file
    # os.remove(os.path.join(data_dir, f"counts_indptr_{partition_idx}.npz"))
# Flush changes to the output file
indptr_cum.flush()


# In[]
# ----------------------------------------------------------------------------
#
# Shuffle the npz file 
#
# ----------------------------------------------------------------------------
data_dir = "/data/zzhang834/hs_healthy_2025_01_30/ordered/"
output_dir = "/data/zzhang834/hs_healthy_2025_01_30/permuted/"
sizes = np.loadtxt(data_dir + "sizes.txt")

meta_cells = []
for partition_idx in range(0, len(sizes)):
    meta_cell_chunk = pd.read_parquet(os.path.join(data_dir, f"obs_{partition_idx}.parquet"))
    meta_cells.append(meta_cell_chunk)
meta_cells = pd.concat(meta_cells, axis = 0)

# permute the index
print("permute the meta data")
np.random.seed(0)
permute_idx = np.random.permutation(meta_cells.shape[0])
chunk_size = 1000000
permute_chunk_idx = [permute_idx[i:i + chunk_size] for i in range(0, len(permute_idx), chunk_size)]
print(f"Total number of chunks: {len(permute_chunk_idx)}")

# 1. permute the meta cells
for cum_idx, chunk_idx in enumerate(permute_chunk_idx):
    meta_cells_chunk = meta_cells.iloc[chunk_idx, :]
    meta_cells_chunk.to_parquet(os.path.join(output_dir, f"obs_{cum_idx}.parquet"))
    del meta_cells_chunk

del meta_cells

print("permute the count matrix")
# for concatenation, calculate the merged size
sizes_new = [0, 0, 0]
for partition_idx in range(0, len(sizes)):
    data_size = sizes[partition_idx, 0]
    indices_size = sizes[partition_idx, 1]
    indptr_size = sizes[partition_idx, 2]
    if partition_idx > 0:
        indptr_size -= 1
    sizes_new[0] += data_size
    sizes_new[1] += indices_size
    sizes_new[2] += indptr_size
data_cum = np.memmap(os.path.join(data_dir, f"counts_data_cum.npz"), dtype = "float32", mode = "r", shape = (int(sizes_new[0]),))
indices_cum = np.memmap(os.path.join(data_dir, f"counts_indices_cum.npz"), dtype = "int16", mode = "r", shape = (int(sizes_new[1]),))
indptr_cum = np.memmap(os.path.join(data_dir, f"counts_indptr_cum.npz"), dtype = "uint64", mode = "r", shape = (int(sizes_new[2]),))

sizes_permute = []
for cum_idx, chunk_idx in enumerate(permute_chunk_idx):
    print(cum_idx)

    # calculate the size of each memmap matrix
    print("calculate the chunk sizes")
    data_chunk_size = 0
    indices_chunk_size = 0
    indptr_chunk_size = 1
    for row in chunk_idx:
        # start and end pointer in indices and data that stores row data
        start, end = indptr_cum[row], indptr_cum[row + 1]
        if (end - start) > 1e5:
            assert False 
        data_chunk_size += (end - start)
        indices_chunk_size += (end - start)
        indptr_chunk_size += 1

    print("save data chunk")
    data_chunk = np.memmap(os.path.join(output_dir, f"counts_data_{cum_idx}.npz"), dtype = 'float32', mode = 'w+', shape = (int(data_chunk_size),))
    ptr = 0
    for row in chunk_idx:
        # start and end pointer in indices and data that stores row data
        start, end = indptr_cum[row], indptr_cum[row + 1]
        data_chunk[ptr:int(ptr + end - start)] = data_cum[start:end]
        ptr += int(end - start)
    data_chunk.flush()

    print("save indices chunk")
    indices_chunk = np.memmap(os.path.join(output_dir, f"counts_indices_{cum_idx}.npz"), dtype = 'int16', mode = 'w+', shape = (int(indices_chunk_size),))
    ptr = 0
    for row in chunk_idx:
        # start and end pointer in indices and data that stores row data
        start, end = indptr_cum[row], indptr_cum[row + 1]
        indices_chunk[ptr:int(ptr + end - start)] = indices_cum[start:end]
        ptr += int(end - start)
    indices_chunk.flush()

    print("save indptr chunk")
    indptr_chunk = np.memmap(os.path.join(output_dir, f"counts_indptr_{cum_idx}.npz"), dtype = 'uint64', mode = 'w+', shape = (int(indptr_chunk_size),))
    ptr = 0
    indptr_chunk[ptr] = 0
    for row in chunk_idx:
        ptr += 1
        # start and end pointer in indices and data that stores row data
        start, end = indptr_cum[row], indptr_cum[row + 1]
        indptr_chunk[ptr] = indptr_chunk[ptr-1] + len(data_cum[start:end])
        assert indptr_chunk[ptr] >= 0
        assert indptr_chunk[ptr] > (indptr_chunk[ptr] - 1)
    indptr_chunk.flush()

    sizes_permute.append([data_chunk.shape[0], indices_chunk.shape[0], indptr_chunk.shape[0]])

    # save into segmentate file
    np.savetxt(os.path.join(output_dir, "sizes.txt"), np.array(sizes_permute))
    del data_chunk, indices_chunk, indptr_chunk    
    print("Done.")



# In[]
# import torch.nn as nn
# import torch.nn.functional as F

# n_mgene = 256
# gene_embed_dict = torch.load(f"dataset/cellxgene_full/gene_embed_meta{n_mgene}_fuzzy.pt")
# compression_mask = torch.zeros((gene_embed_dict["fuzzy_labels"].shape[0], n_mgene))

# for i, neighbors in enumerate(gene_embed_dict["fuzzy_labels"].values):
#     compression_mask[i, neighbors] = 1


# # use a sample dataset for example
# adata = anndata.read_h5ad(data_dir + "blood/partition_0.h5ad")
# adata.var.index = adata.var["feature_id"].values
# adata = adata[:, gene_embed_dict["fuzzy_labels"].index.values]

# # NOTE: this part should be loaded use memmap
# counts = torch.tensor(adata.X.toarray())
# # counts = np.memmap()



# compression_model = CompressionModel(compression_mask = compression_mask)
# counts_meta = compression_model(counts)
# # TODO: transform into sentences
# # feature sentence: gene names, all the same across cells without padding
# # cls, gene1, gene2, gene3, ..., genem
# # expression sentence: torch.cat([0, counts])
# loss = counts_meta.sum(dim = 1).mean()
# loss.backward()

# # print("Parameter matrix:\n", compression_model.compression_proj)
# print("project matrix:\n", compression_model.softmax_act(compression_model.compression_proj * compression_model.compression_mask))
# print("\nGradient of the parameter matrix:\n", compression_model.compression_proj.grad)

# # %%
