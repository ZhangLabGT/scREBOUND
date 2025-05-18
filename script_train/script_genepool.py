# In[]
import anndata
import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt

# In[]
# ------------------------------------------------------------------------------------------
# 1. read in the esm embedding of all human genes
data_dir = "/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/"
gene_embed = torch.load(data_dir + "gene_embed_esm.pt", weights_only = False)
# read in one meta-gene info
gene_meta = anndata.read_h5ad(data_dir + "blood/partition_0.h5ad").var
gene_meta.index = gene_meta["feature_id"].values
print(f"total number of genes in esm model: {len(gene_embed)}")


# In[]
# ------------------------------------------------------------------------------------------
# make sure the gene order matches the sequencing data
geneid = np.loadtxt("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/geneid_select.txt", dtype = object)
gene_embed = torch.vstack([gene_embed[x] for x in geneid])
print(f"total number of overlapping genes: {len(gene_embed)}")

# In[]
# ------------------------------------------------------------------------------------------
# 3. build the k nearest neighbor graph between genes
pca_op = PCA(n_components = 100)
gene_embed_pca = pca_op.fit_transform(StandardScaler().fit_transform(gene_embed.numpy()))
n_mgene = 256

# In[]
import sys
sys.path.append("/project/zzhang834/LLM_KD/src")
import graph_pool
import importlib
importlib.reload(graph_pool)
from data_utils import set_seed

# In[]
device = torch.device("cuda:0")
A = graph_pool.knn_graph(torch.tensor(gene_embed_pca, dtype = torch.float32), k = 30).to(device)
# init spectral clustering
D = torch.diag(A.sum(dim=1))
L = D - A
D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-8))
L_norm = D_inv_sqrt @ L @ D_inv_sqrt
eigvals, eigvecs = torch.linalg.eigh(L_norm)
n_meta = 256
Z = eigvecs[:, 1:n_meta+1]
# 6. KMeans on spectral embedding
labels = KMeans(n_clusters=n_meta, n_init=10, random_state = 0).fit_predict(Z.cpu().numpy())
labels = pd.DataFrame(columns = ["labels"], index = geneid, data = labels[:,None])
labels["feature_name"] = np.array([gene_meta.loc[x, "feature_name"] for x in geneid])

# In[]
_, cluster_size = np.unique(labels["labels"].values.squeeze(), return_counts = True)
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
_ = ax.hist(cluster_size, bins = 50)

# In[]
# Transform into S init
S_init = 0 * torch.ones([A.shape[0], n_meta]) 
for i, label in enumerate(labels["labels"].values.squeeze()):
    S_init[i, label] = 1

# calculate the initial mincut loss
cut = torch.trace(S_init.T @ A.to("cpu") @ S_init)
D = torch.diag(A.sum(dim=1)).to("cpu")
assoc = torch.trace(S_init.T @ D @ S_init)
mincut_loss = -cut / assoc

S_norm = S_init/S_init.pow(2).sum(0, keepdim = True).sqrt()
SS_norm = S_norm.T @ S_norm

# SS_norm = SS / (torch.norm(S_init, p='fro')**2)
I = torch.eye(SS_norm.size(0))
orth_loss = torch.norm(SS_norm - I, p='fro')

print(mincut_loss.item())
print(orth_loss.item())

# In[]
temp = 0.2 
gpool = graph_pool.GenePoolVanilla(A = A.to(device), n_meta = n_meta, s_init = S_init, temp = temp).to(device)
optimizer = optim.Adam(gpool.parameters(), lr = 1e-2)
# place holder
expr_fake = torch.zeros((5, A.shape[0]))

# In[]
n_iters = 50
for it in range(n_iters):
    S, *_ = gpool(gene_embed = gene_embed.to(torch.float32).to(device), expr = expr_fake.to(device))
    # if it == 0:
    #     S_init = S.clone()
    L_c, L_o = graph_pool.mincut_loss(gpool.A, S, add_orthogonality = True)
    loss = L_c + 0.01 * L_o
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Iteration: {it:d}, Loss total: {loss.item():.2f} Loss mincut: {L_c.item():.2f}, Loss ortho: {L_o.item():.2f}")
    # print(gpool.S.grad.sum())

print(S.max(dim = 1).values)
print(S.min(dim = 1).values)

# In[]
# Store gpool for foundation model training
set_seed(0)
cls_embed = torch.randn(1, gene_embed.shape[1])
gene_embed_dict = {"gene_embed": gene_embed, "labels": labels, "gpoolvanilla": gpool.state_dict(), "temp": temp, "token_graph": A.detach().cpu(), "cls_embed": cls_embed}
torch.save(gene_embed_dict, f"/localscratch/ziqi/hs_download/gene_embed_meta{n_mgene}_gpool.pt")

# In[]
EVAL_DATA_DIR = "/project/zzhang834/LLM_KD/dataset/scIB/"
# adata_test = anndata.read_h5ad(EVAL_DATA_DIR + "Immune_ALL_human.h5ad")
# adata_test.obs["label"] = adata_test.obs["final_annotation"]

adata_test = anndata.read_h5ad(EVAL_DATA_DIR + "human_pancreas_norm_complexBatch.h5ad")
adata_test.obs["label"] = adata_test.obs["celltype"]
adata_test.obs["batch"] = adata_test.obs["tech"]

# adata_test = anndata.read_h5ad(EVAL_DATA_DIR + "Lung_atlas_public.h5ad")
# adata_test.obs["label"] = adata_test.obs["cell_type"]

# need to make sure it is the raw count
adata_test.X = adata_test.layers["counts"].copy()

geneid = labels["feature_name"].values
gene_list_common = np.intersect1d(geneid, adata_test.var.index.values.squeeze())
counts = pd.DataFrame(np.zeros((adata_test.shape[0], len(geneid))), index = adata_test.obs.index.values, columns = geneid)
counts.loc[:, gene_list_common] = adata_test[:, gene_list_common].X.toarray()
counts = counts.values.astype(np.float32)

counts_meta = counts @ S.detach().cpu().numpy()
counts_meta = np.log1p(counts_meta/(np.sum(counts_meta, axis = 1, keepdims = True)  +1e-4) * 10e4)

adata_test_meta = anndata.AnnData(X = counts_meta, obs = adata_test.obs)

# In[]
import scanpy as sc
import umap
import utils
# sc.pp.pca(adata_test_meta)
sc.pp.neighbors(adata_test_meta, n_neighbors = 15)
sc.tl.umap(adata_test_meta, min_dist = 0.3)
# X_umap = umap.UMAP(n_neighbors = 15).fit_transform(counts_meta)
colormap =plt.cm.get_cmap("tab20")
fig = utils.plot_embeds(embed = adata_test_meta.obsm["X_umap"], annos = adata_test_meta.obs[["label", "batch"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)

# %%
