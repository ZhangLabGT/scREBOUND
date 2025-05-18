"""
Dataloaders

"""
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
# import gc
import random
# from math import ceil, floor
import anndata
# import scanpy as sc

import os
import pandas as pd
from tqdm import tqdm

def set_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)           # If using GPU
    torch.cuda.manual_seed_all(seed)       # If using multi-GPU
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for Python random
    random.seed(seed)
    # Ensure deterministic behavior in some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ----------------------------------------------------------------------------------------------------------
#
# Ver. 1. Training data are saved in a distributed format, more flexible for training 
# 
# ----------------------------------------------------------------------------------------------------------

def slice_csr(data, indices, indptr, start_row, end_row, num_cols):
    """\
    Description:
    -------------
        Slicing csr matrix by row, used when the csr matrix are saved by (data, indices, indptr) separately
        memory efficient when data, indices, and indptr are saved in memmap format, do not need to load all
    
    Parameters:
    -------------
        data: 1D array (memmaped) that save the data of csr matrix
        indices: 1D array that save the indices of csr matrix
        indptr: 1D array that save the indptr of csr matrix
        start_row: start row of the slicing
        end_row: end row of the slicing
        num_cols: number of columns in csr matrix

    Returns:
    ------------
        csr matrix slice
    """
    
    # Create a new indptr for the slice
    new_indptr = indptr[start_row:end_row+1] - indptr[start_row]
    
    # Slice the data and indices corresponding to the rows
    data_start = indptr[start_row]
    data_end = indptr[end_row]
    
    new_data = data[data_start:data_end]
    new_indices = indices[data_start:data_end]
    
    # Number of rows in the new matrix
    num_rows = end_row - start_row
    
    # Create the new sliced CSR matrix
    try:
        sliced_csr = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(num_rows, num_cols))
    except:
        raise ValueError("issue with index")

    return sliced_csr



class sc_partition(data.Dataset):

    def __init__(self, data_path, min_chunksize, normalize, batch_feats):
        """
        Description:
        -------------
            Create training dataset from distributed training data. Training data are saved by partitions in csr format.
            To save computational resources, the sampling of dataset is by contiguous chunk (of size min_chunksize), so the training data need to be permuted in advance
            If min_chunksize == 1, the sampling of dataset is completely random with more cost.

            NOTE: the current version permute the order of mini-batch within the training chunks (by batch_size), but not permuting the data that consist of each mini-batch
            It is fine with 1 epoch since we already permute the data in advance, but for more epochs, more detailed permutation is necessary to change the composition of each mini-batch.
            Currently the training ordering only change but the composition of each mini-batch is still the same

        Parameters:
        -------------
            data_path: stores the path to the training dataset. Under data_path, there should be count partition files, and meta-data partition files,
            batch_size: the size of each loading data chunks/mini-batch (by number of cells)
            normalize: boolean vector indicating whether the data need to be log-normalized (raw) or not (normalized)

        """
        super(sc_partition, self).__init__()
        
        self.data_path = data_path
        # NOTE: Load the data size file if it exists, or calculate from file size
        if os.path.exists(self.data_path + "sizes.txt"):
            self.sizes = np.loadtxt(self.data_path + "sizes.txt")
        else:
            self.sizes = None
        self.min_chunksize = min_chunksize

        self.normalize = normalize

        # load the batch features
        self.batch_feats_cont = batch_feats["conts"] if batch_feats is not None else None
        self.batch_feats_cat = batch_feats["cats"] if batch_feats is not None else None



    def load_partition(self, idx, label_colname = None, batch_colname = None, data_prefix = "counts", meta_prefix = "obs", save_memory = False):
        """\
        Description:
        -------------
            loading training data partition given partition index

        Parameters:
        -------------
            idx: the index of the training data partition
            label_colname: the column name of cell type label (NOTE: factorized) within the meta-data
            batch_colname: the column name of batch label (NOTE: factorized) within the meta-data
            data_prefix: the prefix of the dataset partition name; for partition {idx}, the csr file name follows `{data_prefix}_data_{idx}.npz', `{data_prefix}_indices_{idx}.npz', `{data_prefix}_indptr_{idx}.npz'
            meta_prefix: the prefix of the meta-data; for partition {idx}, the csr file name follows `{meta_prefix}_{idx}.parquet'
             
        """
        meta_cells = pd.read_parquet(self.data_path + f"{meta_prefix}_{idx}_batchcode.parquet")
        vars = pd.read_csv(self.data_path + "var.csv", index_col = 0)

        self.ncells = meta_cells.shape[0]
        self.ngenes = vars.shape[0]

        fname_expr_data = self.data_path + f"{data_prefix}_data_{idx}.npz"
        fname_expr_indices = self.data_path + f"{data_prefix}_indices_{idx}.npz"
        fname_expr_indptr = self.data_path + f"{data_prefix}_indptr_{idx}.npz"
        # loading the sizes of data, indices, and indptr of each partition
        if self.sizes is not None:
            data_size = self.sizes[idx, 0]
            indices_size = self.sizes[idx, 1]
            indptr_size = self.sizes[idx, 2]
        else:
            data_size = os.path.getsize(fname_expr_data) // np.dtype(np.float32).itemsize
            indices_size = os.path.getsize(fname_expr_indices) // np.dtype(np.int16).itemsize
            indptr_size = os.path.getsize(fname_expr_indptr) // np.dtype(np.uint64).itemsize

        self.expr_data = np.memmap(fname_expr_data, dtype = "float32", mode = "r", shape = (int(data_size), ))
        self.expr_indices = np.memmap(fname_expr_indices, dtype = "int16", mode = "r", shape = (int(indices_size), ))
        self.expr_indptr = np.memmap(fname_expr_indptr, dtype = "uint64", mode = "r", shape = (int(indptr_size), ))

        if save_memory:
            # save reading io but consume more memory
            self.expr_data = np.array(self.expr_data)
            self.expr_indices = np.array(self.expr_data)
            self.expr_indptr = np.array(self.expr_indptr)

        self.batch_ids = meta_cells[batch_colname].values.squeeze()

        if label_colname is not None:
            self.labels = meta_cells[label_colname].values.squeeze()
        else:
            self.labels = None
            

    def __len__(self):
        # Return the number of batches
        return (self.ncells + self.min_chunksize - 1) // self.min_chunksize
    
    def __getitem__(self, idx):
        
        # NOTE: obtain the data mini-batch (start_idx:end_idx) from the training chunk on disk and load it into the memory
        start_idx = int(idx * self.min_chunksize)
        end_idx = int(min((idx + 1) * self.min_chunksize, self.ncells))
        counts = torch.tensor(slice_csr(data = self.expr_data, indices = self.expr_indices, indptr = self.expr_indptr,
                                   start_row = start_idx, end_row = end_idx, num_cols = self.ngenes).astype(np.float32).toarray())
    
        if self.normalize:
            # normalize the raw count and log-transform
            counts_norm = counts/(counts.sum(dim = 1, keepdim = True) + 1e-4) * 10e4
            counts_norm = torch.log1p(counts_norm)
        else:
            counts_norm = counts

        sample = {"counts_norm": counts_norm}

        sample["batch"] = self.batch_ids[start_idx:end_idx]
        if self.batch_feats_cat is not None:
            sample["batch_cat"] = torch.tensor(self.batch_feats_cat[sample["batch"], :], dtype = torch.float32)
        if self.batch_feats_cont is not None:
            sample["batch_cont"] = torch.tensor(self.batch_feats_cont[sample["batch"], :], dtype = torch.float32)
        sample["batch"] = torch.tensor(sample["batch"], dtype = torch.int32)

        if self.labels is not None:
            label = self.labels[start_idx:end_idx]
            sample["label"] = torch.tensor(label, dtype = torch.int32)

        return sample


# ------------------------------------------------------------------------------------------------------------
#
# Evaluation of training result
#
# ------------------------------------------------------------------------------------------------------------

def align_genes(adata, gene_list):
    gene_list_common = np.intersect1d(gene_list, adata.var.index.values.squeeze())
    X = pd.DataFrame(np.zeros((adata.shape[0], len(gene_list))), index = adata.obs.index.values, columns = gene_list)
    X.loc[:, gene_list_common] = adata[:, gene_list_common].X.toarray()

    adata_align = anndata.AnnData(sp.csr_matrix(X.values))
    adata_align.var = pd.DataFrame(index = X.columns)
    adata_align.obs = adata.obs
    return adata_align


def align_genes_memeff(adata, gene_align):
    """
    Memory efficient version: might be slower
    """
    gene_orig = adata.var.index.values.squeeze()
    gene_common = np.intersect1d(gene_align, gene_orig)
    gene_orig_common_position = np.array([np.where(gene_orig == x)[0][0] for x in gene_common])
    gene_align_common_position = np.array([np.where(gene_align == x)[0][0] for x in gene_common])

    counts_align = sp.lil_matrix((adata.shape[0], len(gene_align)))
    for idx in tqdm(range(len(gene_common))):
        counts_align[:, gene_align_common_position[idx]] = adata.X[:, gene_orig_common_position[idx]]

    adata_align = anndata.AnnData(X = counts_align.tocsr())
    adata_align.obs = adata.obs.copy()
    adata_align.var.index = gene_align
    
    return adata_align

class sc_dataset_anndata(data.Dataset):
    """
    construct scdataset from anndata
    """
    def __init__(self, adata, gene_list, batch_feats = None, label_colname = None, batch_colname = None, batch_size = 128, normalize = True):
        """
        expr_path: stores the path to the expr data on disk
        gene_path: stores the path to the gene name of cells on disk 
        meta_path: stores the path to the meta data of cells on disk
        """
        super(sc_dataset_anndata, self).__init__()

        self.ncells = adata.shape[0]

        # check if the count matrix in adata in compressed format
        if isinstance(adata.X, np.ndarray):
            adata.X = sp.csr_matrix(adata.X)

        # find overlapping genes
        if gene_list is not None:
            adata = align_genes(adata, gene_list)
        X = adata.X.toarray()

        # normalize the count
        if normalize:
            libsize = X.sum(axis = 1)
            self.X_norm = X/(libsize[:, None] + 1e-4) * 10e4
            self.X_norm = np.log1p(self.X_norm).astype(np.float32)  
        else:
            self.X_norm = X.astype(np.float32)   

        if batch_feats is not None:
            # load the batch features 
            self.batch_feats_cont = batch_feats["conts"]
            self.batch_feats_cat = batch_feats["cats"]

        self.batch_ids = adata.obs[batch_colname].values.squeeze()

        # note, should be integer values
        if label_colname is not None:
            self.labels = adata.obs[label_colname].values.squeeze()
        else:
            self.labels = None

        self.batch_size = batch_size


    def __len__(self):
        # Return the number of batches
        return (self.ncells + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = int(idx * self.batch_size)
        end_idx = int(min((idx + 1) * self.batch_size, self.ncells))
        # to be decided
        counts_norm = torch.tensor(self.X_norm[start_idx:end_idx,:])

        sample = {"counts_norm": counts_norm}
        if self.labels is not None:
            label = self.labels[start_idx:end_idx]
            sample["label"] = torch.tensor(label, dtype = torch.int32)

        sample["batch"] = self.batch_ids[start_idx:end_idx]
        if self.batch_feats_cat is not None:
            sample["batch_cat"] = torch.tensor(self.batch_feats_cat[sample["batch"], :], dtype = torch.float32)
        if self.batch_feats_cont is not None:
            sample["batch_cont"] = torch.tensor(self.batch_feats_cont[sample["batch"], :], dtype = torch.float32)
        sample["batch"] = torch.tensor(sample["batch"], dtype = torch.int32)

        return sample




