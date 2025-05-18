# In[]
from pathlib import Path
import sys, os

import numpy as np
import pandas as pd

import torch
from torch.utils import data
# from transformers import AdamW
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# sys.path.append("/project/zzhang834/LLM_KD/src")
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")

import data_utils
# from transformer_batch import TransformerModel, get_default_config
# import trainer_batch as trainer_batch
from screbound import TransformerModel, get_default_config
import trainer as trainer_batch

from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention import sdpa_kernel, SDPBackend

def initialize_services(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

# In[]
data_utils.set_seed(0)
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
data_dir = "/data/zzhang834/hs_download/"
# data_dir = "/data/zzhang834/hs_healthy_2025_01_30/"
# Define the device
assert torch.cuda.is_available(), "Training on CPU is not supported"
device = torch.device("cuda")
print(f"GPU - Using device: {device}")

n_mgene = 256
model_config = get_default_config()
batch_size = 64 * 8 * 6
# batch_size = 128
# classifier for 4 gpus, 0.5e-5 too large for less than 4, slightly larger for bp16
lr = 0.3e-5 * (batch_size/32/8) # adjusted for 4 gpus

# accuracy almost the same as fp32
PRECISION = torch.float32
batch_name = "level2"

model_config.__dict__.update({"batch_size": batch_size,
                                "n_epoch": 1,
                                "lr": lr, # important for hyper-parameter tuning
                                "d_embed": 512,
                                "n_head": 8,
                                "d_hidden": 2048, 
                                "n_layer": 6,
                                "d_output": 128,
                                "dropout": 0.1, 
                                "dynamic_maskprob": True, 
                                "lamb_mincut": 1,
                                "mlm_type": "meta",
                                "lamb_mlm": 100,
                                "mask_batchfactor": False, # True,
                                "batch_enc": "cat_concat",
                                "insert_transformer": False,
                                "sup_type": None,
                                "lamb_sup": 0, 
                                "precision": PRECISION,
                                "pretrain_path": None,
                                "checkpoint_path": PROJECT_DIR + "screbound/",
                                "checkpoint_prefix": f"cp_6_512_128_meta",
                                "lognorm_data": False
                                })


token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + f"meta_data/label_dict.pt", weights_only = False)

# ------------------------------------------ batch dict --------------------------------------------------------------------------------
# # original batch feats
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt", weights_only = False)
# # drop the stats features (not very useful)
# batch_dict["cats"] = batch_dict["cats"].drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
# batch_dict["n_cat_list"] = batch_dict["n_cat_list"][4:]

# make value continuous
# batch_feats = pd.read_csv(data_dir + f"meta_data/feature_batch_level2_filter.csv", index_col = 0)
# batch_dict["cats"] = batch_feats[batch_dict["cats"].columns]

# new full list
batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_10.pt", weights_only = False)

# new adaptive batching
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_expr10.pt", weights_only = False)
# ------------------------------------------------------------------------------------------------------------------------------------
batch_dict["cats"] = torch.tensor(batch_dict["cats"].values)

model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device)

# data information
num_partitions = 55
partition_size = 1000000
# remove the last partition as it is for validation
steps_per_partition = np.ceil(partition_size/model_config.batch_size/1) # 489
# the partitions are complete
steps_per_epoch = int((num_partitions-1) * steps_per_partition)
print(f"total number of steps: {steps_per_epoch:d}")

# init optimizer and scheduler, learning rate scale with the batch_size, larger eps for more stability in bf16
optimizer = AdamW(model.parameters(), lr = model_config.lr, eps = 1e-6)
scheduler = OneCycleLR(optimizer, max_lr = model_config.lr, steps_per_epoch = steps_per_epoch, epochs = model_config.n_epoch, pct_start = 0.3)
if model.model_config.dynamic_maskprob:
    model.mask_scheduler = trainer_batch.MaskScheduler(initial_prob = 0.15, final_prob = 0.4, total_steps = steps_per_epoch)
    model.model_config.mask_prob = model.mask_scheduler.get_prob()   

# Load latest checkpoint
if model_config.pretrain_path is not None: 
    print(f"GPU - Preloading lastest model'")
    # load parameter from last train
    state = torch.load(model_config.pretrain_path, weights_only = False)
    # Get the common keys between the current model and the saved model
    filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in model.state_dict()}
    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict, strict = False)

    # NOTE: for continuous training, update optimizer and scheduler for consistent training
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    initial_epoch = state['epoch']
    initial_step = state['step'] + 1
    del state
else:
    initial_epoch = 0
    initial_step = 0
    # If we couldn't find a model to preload, just start from scratch
    print(f'GPU - Could not find model to preload. Starting from scratch')


# Init logger process, only main thread
writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 

# In[]
train_config = {"DIR": data_dir + "permuted/",
                "num_partitions": num_partitions,
                "data_prefix": "counts",
                "meta_prefix": "obs",
                "batch_dict": batch_dict,
                "label_colname": "label_id",
                "batch_colname": "batch_" + batch_name + "_id", 
                "initial_epoch": initial_epoch,
                "initial_step": initial_step,
                "log_step": 100}

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    trainer_batch.train_singlegpu(model = model, train_config = train_config, writer = writer, optimizer = optimizer, scheduler = scheduler)


                




# %%
