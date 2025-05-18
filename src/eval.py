import scib
import pandas as pd
import scanpy as sc
import numpy as np

def eval_batch_correction(adata, embed_key, batch_key, label_key):
    sc.pp.neighbors(adata, use_rep = embed_key)
    # cluster_key is added as the optimal cluster label 
    print("optimal clustering...")
    scib.metrics.cluster_optimal_resolution(adata, cluster_key = "cluster", label_key = label_key)
    print("score calculation...")
    ari_score = scib.metrics.ari(adata, cluster_key = "cluster", label_key = label_key)
    nmi_score = scib.metrics.nmi(adata, cluster_key = "cluster", label_key = label_key)
    asw_score = scib.metrics.silhouette(adata, label_key = label_key, embed = embed_key)
    asw_batch_score = scib.metrics.silhouette_batch(adata, batch_key = batch_key, label_key = label_key, embed = embed_key)

    scores = pd.DataFrame(columns = ["ari", "nmi", "asw", "asw (batch)"], data = np.array([[ari_score, nmi_score, asw_score, asw_batch_score]]))
    return scores
