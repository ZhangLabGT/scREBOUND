# In[]
from pathlib import Path
import sys, os

import numpy as np
import pandas as pd
from datetime import timedelta

# packages for distributed training
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
# from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

sys.path.append("/project/zzhang834/LLM_KD/src")

import data_utils
from screbound import TransformerModel, get_default_config
import trainer

from torch.utils.tensorboard import SummaryWriter

from torch.nn.attention import sdpa_kernel, SDPBackend


def initialize_services(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

# In[]
def main():
    data_utils.set_seed(0)
    PROJECT_DIR = "/project/zzhang834/LLM_KD/"
    data_dir = "/project/zzhang834/hs_download/"

    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {local_rank} - Using device: {device}")
    # Load the dataset
    print(f"GPU {local_rank} - Loading dataset...")

    n_mgene = 256
    model_config = get_default_config()
    batch_size = 64 * 8 * 2
    # classifier for 4 gpus, 0.5e-5 too large for less than 4, slightly larger for bp16
    lr = 0.3e-5 * (batch_size/32/2)

    # accuracy almost the same as fp32
    PRECISION = torch.float32
    batch_name = "level2"

    model_config.__dict__.update({"batch_size": batch_size,
                                    "n_epoch": 1,
                                    "lr": lr, # important for hyper-parameter tuning
                                    "d_embed": 512,
                                    "n_head": 8,  # TODO: make 12 head * 64 dimensions
                                    "d_hidden": 2048, 
                                    "n_layer": 4,
                                    "d_output": 256, # 64,
                                    "dropout": 0.1, # important for hyper-parameter tuning
                                    "dynamic_maskprob": True, # mask_prob is dynamically updated from 0.1 to 0.7 during training
                                    "lamb_mincut": 1,
                                    "mlm_type": "meta",
                                    "lamb_mlm": 1,
                                    "mask_batchfactor": False, # Add mask doesn't make a huge difference
                                    "batch_enc": "encoder",
                                    "insert_transformer": False,
                                    "sup_type": None,
                                    "lamb_sup": 0, 
                                    "precision": PRECISION,
                                    "pretrain_path": None,
                                    "checkpoint_path": PROJECT_DIR + "checkpoint/",
                                    "checkpoint_prefix": f"cp_vanilla_4_512_256_meta_enccatadp_{batch_name}",
                                    "lognorm_data": False
                                    })
    
    token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
    label_dict = torch.load(data_dir + f"meta_data/label_dict.pt", weights_only = False)
    # batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt", weights_only = False)
    batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_10.pt", weights_only = False)
    # ------------------------------------------------------------------------------------------------------------------------------------
    # # drop the stats features (not very useful)
    # batch_dict["cats"] = batch_dict["cats"].drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
    # batch_dict["n_cat_list"] = batch_dict["n_cat_list"][4:]

    # new adaptive batching
    batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_expr10.pt", weights_only = False)
    # ------------------------------------------------------------------------------------------------------------------------------------
    batch_dict["cats"] = torch.tensor(batch_dict["cats"].values)

    model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device)

    # wrap model into multi-gpus setting
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # data information
    num_partitions = 55
    partition_size = 1000000
    # remove the last partition as it is for validation
    steps_per_partition = np.ceil(partition_size/model_config.batch_size/num_gpus) # 489
    # the partitions are complete
    steps_per_epoch = int((num_partitions-1) * steps_per_partition)
    if global_rank == 0:
        print(f"total number of steps: {steps_per_epoch:d}")


    # init optimizer and scheduler, learning rate scale with the batch_size, larger eps for more stability in bf16
    optimizer = AdamW(model.parameters(), lr = model_config.lr, eps = 1e-6)
    scheduler = OneCycleLR(optimizer, max_lr = model_config.lr, steps_per_epoch = steps_per_epoch, epochs = model_config.n_epoch, pct_start = 0.3)
    # model related
    if model.module.model_config.dynamic_maskprob:
        model.module.mask_scheduler = trainer.MaskScheduler(initial_prob = 0.15, final_prob = 0.4, total_steps = steps_per_epoch)
        model.module.model_config.mask_prob = model.module.mask_scheduler.get_prob()   


    # Init logger process, only main thread
    if global_rank == 0:
        writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 
    else:
        writer = None

    # sync
    dist.barrier()

    train_config = {"DIR": data_dir + "permuted/",
                    "num_partitions": num_partitions,
                    "data_prefix": "counts",
                    "meta_prefix": "obs",
                    "batch_dict": batch_dict,
                    "label_colname": "label_id",
                    "batch_colname": "batch_" + batch_name + "_id",
                    "initial_epoch": 0,
                    "initial_step": 0,
                    "log_step": 100,
                    "global_rank": global_rank}


    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        trainer.train_multigpus(model = model, train_config = train_config, optimizer = optimizer, scheduler = scheduler, writer = writer)

                    
                    

# In[]
if __name__ == '__main__':

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # environment variables generated when use torchrun
    # local gpu id within the machine
    local_rank = int(os.environ['LOCAL_RANK'])
    # global gpu id across machines (uniq), same as local with one machine
    global_rank = int(os.environ['RANK'])
    print(f"local rank: {local_rank}")
    print(f"global rank: {global_rank}")

    # increase the time-out waiting time across gpus
    init_process_group(backend='nccl', timeout=timedelta(minutes=60))
    torch.cuda.set_device(local_rank) # Set the device to local rank

    main()
    
    destroy_process_group()

# %%
