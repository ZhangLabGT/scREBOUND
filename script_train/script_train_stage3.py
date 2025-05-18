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
batch_size = 64 * 8 * 6
# batch_size = 128
# classifier for 4 gpus, 0.5e-5 too large for less than 4, slightly larger for bp16
lr = 0.3e-5 * (batch_size/32/8) # adjusted for 4 gpus

# accuracy almost the same as fp32
PRECISION = torch.float32
# name of columns used when training
batch_name = "level2"

# new dataset with encoder
# model_name = f"cp_vanilla_4_512_256_enc_{batch_name}_1"
model_name = f"cp_6_512_256_1"
PRETRAIN_MODEL = PROJECT_DIR + "screbound/" + model_name + ".pth"

state = torch.load(PRETRAIN_MODEL, weights_only = False)
model_config = get_default_config()
model_config.__dict__.update(state["model_config"])
# further update
model_config.__dict__.update({"batch_size": batch_size,
                              "lr": lr,
                              "mask_prob": 0.1,
                              "dynamic_maskprob": True, # mask_prob is dynamically updated from 0.1 to 0.7 during training
                              "lamb_mincut": 1,
                              "mlm_type": "raw",
                              "lamb_mlm": 10,
                              "sup_type": "contrcb",
                              "lamb_sup": 1,
                              "precision": PRECISION,
                              "checkpoint_path": PROJECT_DIR + "screbound/",
                              "checkpoint_prefix": model_name.removesuffix("_1") + "_contrcb1_dyn",
                              "lognorm_data": False
                            })

for x, val in model_config.__dict__.items():
    print(x, end = ": ")
    print(val)
    
token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + f"meta_data/label_dict.pt", weights_only = False)
# ------------------------------------------------------------------------------------------------------------------------------------
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt", weights_only = False)
# drop the stats features (not very useful)
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
model.freeze_fm_gradient(freeze_trans = False, freeze_batchenc = False, freeze_compression = False)

# Load latest checkpoint
print(f"GPU - Preloading pretrained model, overwrite gene compression states")
# Get the common keys between the current model and the saved model
filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in model.state_dict()}
# Load the filtered state dictionary into the model
model.load_state_dict(filtered_state_dict, strict = False)

# data information
# old
num_partitions = 55
# new
# num_partitions = 79
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
    model.mask_scheduler = trainer_batch.MaskScheduler(initial_prob = 0.1, final_prob = 0.2, total_steps = steps_per_epoch)
    model.model_config.mask_prob = model.mask_scheduler.get_prob()   

initial_epoch = 0
initial_step = 0

# Init logger process, only main thread
writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 

# In[]
print("Start training stage 3...")
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

print("Done stage 3")
# %%
