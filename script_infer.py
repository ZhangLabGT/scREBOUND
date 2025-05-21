# In[]
import torch
from torch.utils import data
from torch.amp import autocast

import anndata
import argparse

import tqdm
import pandas as pd

import sys, os
sys.path.append("./src")
import data_utils
import batch_encode 
import trainer
import screbound
import yaml

import warnings
warnings.filterwarnings("ignore")


def evaluate_mtp(model, dataloader, mask_prob):
    model.model_config.mask_prob = mask_prob
    # need to remove contr loss
    use_contr = model.model_config.use_contr
    model.model_config.use_contr = False
    
    # NOTE: training loop
    model.eval()
    with torch.no_grad():
        val_loss_mlm = 0.0
        for data_sample in tqdm.tqdm(dataloader, desc=f"Evaluation"):
            with autocast(device_type="cuda", dtype = torch.bfloat16, enabled = model.model_config.use_flashatten):
                if model.model_config.batch_enc == "onehot":
                    del data_sample["batch"]

                loss, loss_item = trainer.infer_databatch(model, data_sample, multigpus = False)                            
            val_loss_mlm += loss_item["mlm"]

        val_loss_mlm /= len(dataloader)
        print(f"Val Loss (MLM): {val_loss_mlm:.4f}")

    model.model_config.use_contr = use_contr

    return val_loss_mlm

# In[]
# TODO: add parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", default = "./infer_config.yaml", help="Path to config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)


# function
data_utils.set_seed(config["seed"])
device = torch.device(config["device"])
print(f"GPU - Using device: {device}")

state = torch.load(config["model_dir"], weights_only = False)

model_pretrain = trainer.load_model(state = state, device = device)

# In[]
# data_dir = "/net/csefiles/xzhanglab/zzhang834/scREBOUND/eval_adata/pancreas_aligned.h5ad"
adata_test = anndata.read_h5ad(config["data_dir"])

# output_dir =  f"./results_screbound/"
output_dir = config["output_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# NOTE: Calculate the batch factor
if model_pretrain.model_config.batch_enc is not None:
    print("create batch factor...")
    batch_features = batch_encode.construct_batch_feats(adata = adata_test, use_mito = True, use_tech = False, use_nmeasure = False)
    batch_features_digitize, max_vals = batch_encode.tokenize_batch_feats(batch_features, max_vals = state["model_config"]["batch_enc"]["max_cat_vals"],
                                                                          nbins = state["model_config"]["batch_enc"]["n_cat_list"][0], only_genes = False)    
    batch_features_digitize = torch.tensor(batch_features_digitize.values, dtype = torch.float32)

else:
    batch_features_digitize = None

print("create dataloader...")
label_colname = None
batch_colname = "batch_id"
test_dataset = data_utils.sc_dataset_anndata(adata = adata_test, gene_list = state["token_dict"]["labels"]["feature_name"].values, batch_feats = {"conts": None, "cats": batch_features_digitize},
                                             label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)


# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: calculate the embedding
#
# --------------------------------------------------------------------------------------------------------------

adata_embed = trainer.cell_embed(model = model_pretrain, dataloader = test_loader, multi_gpus = False)
adata_embed.obs = adata_test.obs.copy()
adata_embed.obsm["latent"] = adata_embed.X.copy()

adata_embed.write_h5ad(output_dir + f"adata_embed.h5ad")

# In[]
import utils
import scanpy as sc
import matplotlib.pyplot as plt

print("visualization...")
adata_embed = anndata.read_h5ad(output_dir + f"adata_embed.h5ad")

sc.pp.neighbors(adata_embed, n_neighbors = 30, use_rep = "latent")
sc.tl.umap(adata_embed, min_dist = 0.5)

adata_embed.obsm[f"X_umap_latent"] = adata_embed.obsm["X_umap"].copy()
del adata_embed.obsm["X_umap"]

use_rep = "latent"
colormap =plt.cm.get_cmap("tab20")

figsize = (12, 7)
annos = adata_embed.obs[["label", "batch_id"]].astype("category")
        
fig = utils.plot_embeds(embed = adata_embed.obsm[f"X_umap_{use_rep}"], annos = annos, markerscale = 15, figsize = figsize, s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(output_dir + "embed_umap.png", bbox_inches = "tight")
print("Done.")

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Calculate MTP loss
#
# --------------------------------------------------------------------------------------------------------------
val_loss_mtp_list = []
mask_prob_list = [0.1, 0.2, 0.3, 0.4]
for mask_prob in mask_prob_list:
    val_loss_mtp = evaluate_mtp(model_pretrain, test_loader, mask_prob = mask_prob)
    val_loss_mtp_list.append(val_loss_mtp)
    
val_loss_df = pd.DataFrame(columns = ["mask prob", "val mtp"])
val_loss_df["val mtp"] = val_loss_mtp_list
# val_loss_df.to_csv(res_dir + f"{data_case}_valloss.csv")


# %%
