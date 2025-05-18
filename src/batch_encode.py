import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

mito_gene_name = ['MT-ND6', 'MT-CO2', 'MT-CYB', 'MT-ND2', 'MT-ND5', 'MT-CO1', 'MT-ND3', 'MT-ND4', 'MT-ND1', 'MT-ATP6', 'MT-CO3', 'MT-ND4L', 'MT-ATP8']

# selected in batch_feature_filtered.csv, see the batch_preprocess script for the greedy selection algorithm
hk_gene_name = ['AP2M1', 'BSG', 'CD59', 'CSNK2B', 'EDF1', 'EEF2', 'GABARAP', 'HNRNPA2B1', 'HSP90AB1', 'MLF2', 'MRFAP1', 'PCBP1', 'PFDN5', 'PSAP', 'RAB11B', 'RAB1B', 'RAB7A', 'RHOA', 'UBC']
ribo_gene_name = ['RPS18', 'RPL41', 'RPL27A', 'RPL5', 'RPS26', 'RPL36A', 'RPS17', 'RPS10', 'RPL17', 'RPS4Y1', 'RPS6KA2']
stress_gene_name = ['ALK', 'APOD', 'BCL2', 'GAD2', 'MCL1', 'PPP3CB']

batch_gene_name = hk_gene_name + ribo_gene_name + stress_gene_name

assay_codebook = {"10x 3' v3": 0, "Smart-seq2": 1, "10x 3' v2": 2, "10x 5' v1": 3, "10x 5' transcription profiling": 4, "Seq-Well": 5, 
                "10x 3' v1": 6, "10x 5' v2": 7, "Seq-Well S3": 8, "Drop-seq": 9, "microwell-seq": 10, "Smart-seq v4": 11,
                "ScaleBio single cell RNA sequencing": 12, "10x 3' transcription profiling": 13, "TruDrop": 14, "MARS-seq": 15,
                "CEL-seq2": 16, "SPLiT-seq": 17, "BD Rhapsody Whole Transcriptome Analysis": 18, "BD Rhapsody Targeted mRNA": 19,
                "sci-RNA-seq": 20} # , "other": -1
suspension_codebook = {"cell": 0, "nucleus": 1}


def construct_expr_feats(counts_raw, batch_labels, batch_list, gene_name, gene_select = batch_gene_name):

    if isinstance(counts_raw, dict):
        counts_raw = sp.csr_matrix((counts_raw["data"], counts_raw["indices"], counts_raw["indptr"]), shape=(counts_raw["row"], counts_raw["col"]))

    gene_select_position = np.array([np.where(gene_name == x)[0][0] for x in gene_select])
    mito_gene_position = np.array([np.where(gene_name == x)[0][0] for x in mito_gene_name])

    batch_stats = pd.DataFrame(data = 0.0, index = batch_list, columns = ["libsize", "nnz", "raw_mean_nnz", "ncells"])
    expr_batch = pd.DataFrame(data = 0.0, index = batch_list, columns = gene_select)
    prop_mito = pd.DataFrame(data = 0.0, index = batch_list, columns = ["prop_mito"])

    for batch in batch_list:
        batch_idx = (batch_labels == batch)
        if np.sum(batch_idx) == 0:
            continue

        counts_batch = counts_raw[batch_idx, :]

        # calculate the batch_specific expr stats
        libsize = np.array(counts_batch.sum(axis = 1)).flatten()
        # number of non-zero genes for each cell
        nnz = np.array((counts_batch > 0).sum(axis = 1)).flatten()
        # proportion of non-zero genes for each cell, divided by total number of genes
        nnz_prop = nnz/counts_batch.shape[1]
        # total count per cell / total number of non-zero genes per cell
        raw_mean_nnz = libsize/nnz
        batch_stats.loc[batch, "libsize"] += libsize.sum()
        batch_stats.loc[batch, "nnz"] += nnz_prop.sum()
        batch_stats.loc[batch, "raw_mean_nnz"] += raw_mean_nnz.sum()
        batch_stats.loc[batch, "ncells"] += np.sum(batch_idx)

        # calculate the mito proportion
        mito = np.array(counts_batch[:, mito_gene_position].sum(axis = 1)).flatten()/libsize
        prop_mito.loc[batch, "prop_mito"] += mito.sum()

        # calculate the expr of important genes
        expr_batch.loc[batch, gene_select] += np.log1p(counts_batch[:, gene_select_position].toarray()/(libsize[:,None] + 1e-4) * 10e4).sum(0)

    return {"expr": expr_batch, "prop_mito": prop_mito, "batch_stats": batch_stats}    



def construct_batch_feats(adata, use_mito = True, use_tech = False, use_nmeasure = False, norm_batchsize = True):
    
    if use_tech:
        batch_info = ["assay", "suspension_type"]
        assert "assay" in adata.obs.columns
        assert "suspension_type" in adata.obs.columns
    else:
        batch_info = []

    batch_info += ["libsize", "nnz", "raw_mean_nnz"]
    if use_nmeasure:
        batch_info += ["n_measures"]
        assert "n_measures" in adata.obs.columns

    if use_mito:
        batch_info += ["prop_mito"]

    assert "batch_id" in adata.obs.columns
    # calculate the remaining stats
    X = adata.layers["counts"]

    uniq_batches = np.unique(adata.obs["batch_id"].values)
    batch_feats = pd.DataFrame(data = 0, index = uniq_batches, columns = batch_info + batch_gene_name)
    # only for tech information
    for batch in uniq_batches:
        adata_batch = adata[adata.obs["batch_id"] == batch, :]

        if use_tech:
            assay = np.unique(adata_batch.obs["assay"].values)
            try:
                assert len(assay) == 1
            except:
                raise ValueError(f"there should be only one assay type in batch {batch}")
            batch_feats.loc[batch, "assay"] = assay[0]
            suspension_type = np.unique(adata_batch.obs["suspension_type"].values)
            try:
                assert len(suspension_type) == 1
            except:
                raise ValueError(f"there should be only one suspension type in batch {batch}")      
            batch_feats.loc[batch, "suspension_type"] = suspension_type[0]

        if use_nmeasure:
            # n_measure should already be saved in adata
            batch_feats.loc[batch, "n_measures"] = adata_batch.obs["n_measures"].values.mean()

    expr_dict = construct_expr_feats(counts_raw = X, batch_labels = np.array([x for x in adata.obs["batch_id"]]),
                                    batch_list = uniq_batches, gene_name = np.array([x for x in adata.var.index]), gene_select = batch_gene_name)
    
    # TODO: Log-transform feature in tokenize data function
    ncells = expr_dict["batch_stats"]["ncells"].values.squeeze()
    batch_feats.loc[expr_dict["expr"].index, ["libsize", "nnz", "raw_mean_nnz"]] = expr_dict["batch_stats"][["libsize", "nnz", "raw_mean_nnz"]].values/ncells[:, None]
    batch_feats.loc[expr_dict["expr"].index, batch_gene_name] = expr_dict["expr"][batch_gene_name].values/ncells[:, None]

    if use_mito:
        batch_feats.loc[expr_dict["prop_mito"].index, ["prop_mito"]] = expr_dict["prop_mito"].values/ncells[:,None]


    return batch_feats



def tokenize_batch_feats(batch_feats, max_vals = None, margin = 0.2, nbins = 10, normalize = True, only_genes = False):
    """
    Description:
    -------------
        Transform the batch_feats table into the digitized table values
    """

    if only_genes:
        batch_feats_filter = batch_feats[batch_gene_name]
    else:
        # batch_info = ["libsize", "nnz", "raw_mean_nnz", "prop_mito"]
        batch_info = ["nnz", "raw_mean_nnz", "prop_mito"]
        batch_feats_filter = batch_feats[batch_info + batch_gene_name]
        
        # value preprocess, NOTE: libsize, raw_mean_nnz need to be log-transformed
        # log_libsize = np.log1p(batch_feats_filter["libsize"].values.squeeze())
        log_nnz = np.log1p(batch_feats_filter["raw_mean_nnz"].values.squeeze())
        log_prop_mito = np.log1p(batch_feats_filter["prop_mito"].values.squeeze() * 1000)

        # batch_feats_filter.loc[:, "libsize"] = log_libsize
        batch_feats_filter.loc[:, "raw_mean_nnz"] = log_nnz
        batch_feats_filter.loc[:, "prop_mito"] = log_prop_mito

    # determine binsize according to max value of each gene
    if max_vals is None:
        if normalize:
            max_vals = np.max(batch_feats_filter.values, axis = 0) + margin
        else:
            max_vals = np.array([np.max(batch_feats_filter.values) + margin] * batch_feats_filter.shape[1])

    else:
        # upper bound the values, NOTE: no need for margin if cut by upper bound
        for idx, col in enumerate(batch_feats_filter.columns):
            batch_feats_filter.loc[batch_feats_filter[col].values.squeeze() > max_vals[idx], col] = max_vals[idx]
    min_vals = np.zeros_like(max_vals)

    # min-max normalization, easiler for binning
    batch_feats_norm = (batch_feats_filter.values - min_vals[None, :])/(max_vals[None, :] - min_vals[None, :])

    # 10 bins: 9 digits and 0->0 [0, 1, ..., 10], make sure all bins has training data
    bins = np.arange(0, 1, 1/(nbins - 1))
    batch_feats_digit = np.digitize(batch_feats_norm, bins = bins, right = True)    
    batch_feats_digit = pd.DataFrame(index = batch_feats_filter.index, columns = batch_feats_filter.columns, data = batch_feats_digit)
    return batch_feats_digit, max_vals

