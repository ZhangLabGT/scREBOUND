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

import warnings
warnings.filterwarnings("ignore")


def evaluate_mlm(model, dataloader, mask_prob):
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

def reformat_statedict(state, token_dict, label_dict, device, verbose = True):
    # create model configuration profile
    new_dict = {}

    model_config = screbound.get_default_config()

    # correction:
    state["model_config"]['mlm_type'] = "raw"
    
    state['model_config']['checkpoint_path'] = None
    state['model_config']['checkpoint_prefix'] = None
    state['model_config']['use_contr'] = True if state['model_config']['sup_type'] is not None else False
    state['model_config']['lamb_contr'] = state["model_config"]["lamb_sup"]

    if state['model_config']['batch_enc'] is not None:
        state["model_config"]['batch_enc'] = {"name": "cat_concat",
                                              "n_cat_list": len([x for x in state["model_state_dict"].keys() if x.startswith("batch_encoder.enc_cat")]) \
                                              * [state["model_state_dict"]["batch_encoder.enc_cat.0.weight"].shape[0]]
                                                }
        
    del state["model_config"]["mask_batchfactor"], state["model_config"]["insert_transformer"], state["model_config"]["sup_type"], state["model_config"]["lamb_sup"]
    
    model_config.__dict__.update(state["model_config"])

    for x, val in model_config.__dict__.items():
        print(x, end = ": ")
        print(val)
    new_dict["model_config"] = model_config.__dict__
    new_dict['token_dict'] = token_dict.copy()
    new_dict['label_dict'] = label_dict.copy()

    # create model with profile
    model = screbound.TransformerModel(model_config = model_config, token_dict = token_dict, label_dict = label_dict, device = device).to(model_config.precision)

    # load model params
    if verbose:
        print("Load parameters...")

    # correction
    from collections import OrderedDict
    # Replace the prefix in keys
    new_state_dict = OrderedDict()
    for k, v in state["model_state_dict"].items():
        if k.startswith('expr_predictor_restart'):
            k_new = k.replace("expr_predictor_restart", "expr_predictor")
            print(k_new)
            new_state_dict[k_new] = v.clone()
        else:
            new_state_dict[k] = v.clone()

    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
    new_dict['model_state_dict'] = filtered_state_dict.copy()

    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict, strict=False)
    if verbose:
        print("Done.")

    return model, new_dict


# In[]
# TODO: add parser


# function
data_utils.set_seed(0)
device = torch.device("cuda")
print(f"GPU - Using device: {device}")

model_name = f"screbound_batchenc_contr"
# model_name = f"screbound_batchenc"
# model_name = f"screbound_contr"
# model_name = f"screbound_vanilla"
model_dir = f"./model_statedict_clean/{model_name}.pth"
state = torch.load(model_dir, weights_only = False)

token_dict = torch.load("./model_aux/hs_gpool.pt", weights_only = False)
label_dict = torch.load("./model_aux/cxg_hs_0731/label_dict.pt", weights_only = False)
batch_dict = torch.load("./model_aux/cxg_hs_0731/batch_dict.pt", weights_only = False)


model_pretrain = trainer.load_model(state = state, device = device)
# model_pretrain, new_state = reformat_statedict(state = state, token_dict = token_dict, label_dict = label_dict, device = device)
# torch.save(new_state, f = f"./model_statedict_clean/{model_name}.pth")

# In[]
data_dir = "/net/csefiles/xzhanglab/zzhang834/scREBOUND/eval_adata/pancreas_aligned.h5ad"
adata_test = anndata.read_h5ad(data_dir)

output_dir =  f"./results_screbound/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# NOTE: Calculate the batch factor
if model_pretrain.model_config.batch_enc is not None:
    print("create batch factor...")
    batch_features = batch_encode.construct_batch_feats(adata = adata_test, use_mito = True, use_tech = False, use_nmeasure = False)
    batch_features_digitize, max_vals = batch_encode.tokenize_batch_feats(batch_features, max_vals = batch_dict["cat_maxvals"], nbins = 10, only_genes = False)    
    batch_features_digitize = torch.tensor(batch_features_digitize.values, dtype = torch.float32)

else:
    batch_features_digitize = None

print("create dataloader...")
label_colname = None
batch_colname = "batch_id"
test_dataset = data_utils.sc_dataset_anndata(adata = adata_test, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize},
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

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Evaluation MLM task
#
# --------------------------------------------------------------------------------------------------------------
val_loss_list = []
val_loss_mlm_list = []
val_loss_metric_list = []
mask_prob_list = [0.1, 0.2, 0.3, 0.4]
for mask_prob in mask_prob_list:
    val_loss, val_loss_mlm, val_loss_metric,val_loss_mincut, val_loss_ortho = evaluate_mlm(model_pretrain, test_loader, mask_prob = mask_prob)
    val_loss_list.append(val_loss)
    val_loss_mlm_list.append(val_loss_mlm)
    val_loss_metric_list.append(val_loss_metric)

val_loss_df = pd.DataFrame(columns = ["mask prob", "val total", "val mlm", "val metric", "dataset"])
val_loss_df["val mlm"] = val_loss_mlm_list
# val_loss_df.to_csv(res_dir + f"{data_case}_valloss.csv")


# %%
