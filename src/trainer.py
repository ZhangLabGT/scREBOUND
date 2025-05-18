import torch
import numpy as np
import torch.nn as nn

# for cell_embed
import scipy.sparse as sparse
import pandas as pd
import anndata
import tqdm
import math

from torch.utils import data
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# for bf16 training
from torch.amp import autocast, GradScaler

import data_utils
import screbound
import base_model
import contrastive

CAST_DTYPE = torch.bfloat16

def load_model(state, device, verbose = True, hard_assign = True):
    # create model configuration profile
    model_config = screbound.get_default_config()
    model_config.__dict__.update(state["model_config"])
    if verbose:
        for x, val in model_config.__dict__.items():
            print(x, end = ": ")
            print(val)

    # create model with profile
    model = screbound.TransformerModel(model_config = model_config, token_dict = state["token_dict"], label_dict = state["label_dict"], device = device).to(model_config.precision)

    # load model params
    if verbose:
        print("Load parameters...")
    if not hard_assign:
        filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in model.state_dict()}
        # Load the filtered state dictionary into the model
        model.load_state_dict(filtered_state_dict, strict = False)
    else:
        model.load_state_dict(state_dict = state["model_state_dict"])
    if verbose:
        print("Done.")
    return model


class MaskScheduler:
    """
    Mask probability schedular
    """
    def __init__(self, initial_prob=0.15, final_prob=0.4, total_steps=10000):
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_prob(self):
        progress = min(self.current_step / self.total_steps, 1.0)
        prob = self.initial_prob + progress * (self.final_prob - self.initial_prob)

        return prob

def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path, multi_gpus = True):
    if multi_gpus:
        model_acc = model.module
    else:
        model_acc = model
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_acc.state_dict(),
        'model_config': model_acc.model_config.__dict__, # save the model config for repeated training too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")



def infer_databatch(model, data_sample, multigpus: bool = True):
    """
    Description:
    ------------
        Forward pass and loss calculation of the foundation model on input data_sample

    Parameters:
    ------------
        model: the transformer foundation model
        data_sample: the input data sample to the model
        multigpus: boolean value indicating the use of multi-gpus or not

    Return:
    ------------
        loss, loss_dict

    """
    if multigpus:
            model_acc = model.module
    else:
            model_acc = model

    # NOTE: data_sample["counts_norm"] of shape (nchunks, chunksize, nfeats), need to reshape to (batchsize, nfeats)
    # batchsize = nchunks * chunksize, if nchunks == 1, the batch is the same as chunk, not an ideal permutation
    expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)
    batch_sample_id = data_sample["batch"].reshape(-1).to(model_acc.device, non_blocking = True) if "batch" in data_sample.keys() else None
    batch_sample_cat = data_sample["batch_cat"].reshape(-1, data_sample["batch_cat"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cat" in data_sample.keys() else None
    batch_sample_cont = data_sample["batch_cont"].reshape(-1, data_sample["batch_cont"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cont" in data_sample.keys() else None

    all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample)
    expr_pred, expr_pred_meta = model_acc.predict_expr(cell_embed = cell_embed, batch_factors = batch_sample_cat, batch_ids = batch_sample_id)

    if model_acc.model_config.mlm_type == "meta":
        *_, expr_sample_meta = model_acc.gene_compression(gene_embed = model_acc.token_embed, expr = expr_sample, log_norm = True)
        # loss_mlm = ((expr_pred_meta - expr_sample_meta) * mask_gene).pow(2).sum(1).mean()
        loss_mlm = (1/mask_gene.sum(1) * ((expr_pred_meta - expr_sample_meta) * mask_gene).pow(2).sum(1)).mean()
    
    else:
        assert model_acc.model_config.lognorm_data == False
        libsize = expr_sample.sum(1)
        expr_pred_mean = expr_pred["mean"] * libsize[:, None]
        expr_pred_disp = expr_pred["disp"]
        loss_mlm = - (1/mask_gene.sum(1) * (base_model.log_nb_positive(x = expr_sample, mu = expr_pred_mean, theta = expr_pred_disp) * mask_gene).sum(1)).mean()

    # contrastive loss
    if model_acc.model_config.use_contr:
        label_sample_id = data_sample["label"].reshape(-1).to(model_acc.device, non_blocking = True)
        batch_label_contr = batch_sample_id

        # if multigpus: 
        #     contr = contrastive.SupContrLossMultiGPUs(label_asso_mtx = model_acc.contrastive_label_mtx, temperature = 0.07, unknown_label = model_acc.label_unknown)
        # else:
        contr = contrastive.SupContrLoss(label_asso_mtx = model_acc.contrastive_label_mtx, temperature = 0.07)
    
        # remove unknown for better calculation, doesn't work for multi-gpus
        unknown_samples = (label_sample_id == model_acc.label_unknown)
        cell_embed = cell_embed[~unknown_samples]
        label_sample_id = label_sample_id[~unknown_samples]
        batch_label_contr = batch_label_contr[~unknown_samples]
        
        loss_contr = contr(features = cell_embed, label_ids = label_sample_id, batch_ids = batch_label_contr)

    else:
        loss_contr = torch.tensor([0.0], device = model_acc.device)

    # mincut loss
    l_mincut, l_ortho = model_acc.gene_compression.mincut_loss(add_ortho = True)
    
    loss = model_acc.model_config.lamb_mlm * loss_mlm + model_acc.model_config.lamb_mincut * (l_mincut + 0.01 * l_ortho) + model_acc.model_config.lamb_contr * loss_contr

    return loss, {"mlm": loss_mlm.item(), "mincut": l_mincut.item(), "ortho": l_ortho.item(), "metric": loss_contr.item()}


def cell_embed(model, dataloader, multi_gpus = True):
    """
    Description:
    ------------
        Obtain the model cell embedding for data in dataloader
    
    Parameters:
    ------------
        model: the transformer model
        dataloader: the dataloader for the input data
        mask_prob: the masking probability of data in the forward pass, default is 0

    """
    if multi_gpus:
        model_acc = model.module
    else:
        model_acc = model
    
    # evaluation model
    model_acc.eval()
    # update the mask_prob for evaluation
    model_acc.model_config.mask_prob = 0.0

    if model_acc.model_config.use_flashatten:
        # because flashattention only accept 16bit model
        enable_casting = True
    else:
        enable_casting = False

    cell_embeds = []
    cell_embeds_contr = []
    labels = []
    batches = []
    with torch.no_grad():
        for data_sample in tqdm.tqdm(dataloader, desc=f"Calc embed"):

            with autocast(device_type="cuda", dtype = CAST_DTYPE, enabled = enable_casting):
                expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)

                # all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample, batch_factors_cont = batch_sample_cont, batch_factors_cat = batch_sample_cat, batch_ids = None)                    
                all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample)                    
                cell_embeds.append(sparse.csr_matrix(cell_embed.to(torch.float32).detach().cpu().numpy()))  

            if "label" in data_sample.keys():
                labels.append(data_sample["label"].reshape(-1).detach().cpu().numpy())
            else:
                labels.append(np.array([np.nan] * cell_embed.shape[0]))

            if "batch" in data_sample.keys():
                batches.append(data_sample["batch"].reshape(-1).detach().cpu().numpy())
            else:
                batches.append(np.array([np.nan] * cell_embed.shape[0]))

    cell_embeds = sparse.vstack(cell_embeds)
    labels = np.concatenate(labels, axis = 0)
    batches = np.concatenate(batches, axis = 0)
    meta = pd.DataFrame.from_dict({"label_id": labels, "batch_id": batches})
    adata = anndata.AnnData(X = cell_embeds, obs = meta.astype("category"))

    return adata



def cell_impute(model, dataloader, multi_gpus = True, only_mean = True):
    """
    Description:
    ------------
        Obtain the model cell embedding for data in dataloader
    
    Parameters:
    ------------
        model: the transformer model
        dataloader: the dataloader for the input data
        only_mean: return only the mean of imputed data

    """
    if multi_gpus:
        model_acc = model.module
    else:
        model_acc = model
    
    # evaluation model
    model_acc.eval()
    # remove mask
    model_acc.model_config.mask_prob = 0.0

    if model_acc.model_config.use_flashatten:
        # because flashattention only accept 16bit model
        enable_casting = True
    else:
        enable_casting = False

    counts_impute_norm = []
    counts_impute_disp = []
    labels = []
    batches = []
    with torch.no_grad():
        for data_sample in tqdm.tqdm(dataloader, desc=f"Calc embed"):

            with autocast(device_type="cuda", dtype = CAST_DTYPE, enabled = enable_casting):
                expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)
                batch_sample_id = data_sample["batch"].reshape(-1).to(model_acc.device, non_blocking = True) if "batch" in data_sample.keys() else None
                batch_sample_cat = data_sample["batch_cat"].reshape(-1, data_sample["batch_cat"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cat" in data_sample.keys() else None
                batch_sample_cont = data_sample["batch_cont"].reshape(-1, data_sample["batch_cont"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cont" in data_sample.keys() else None

                all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample)
                expr_pred, expr_pred_meta = model_acc.predict_expr(cell_embed = cell_embed, batch_factors = batch_sample_cat)

                # assert model_acc.model_config.mlm_type == "raw"
                assert model_acc.model_config.lognorm_data == False

            counts_impute_norm.append(expr_pred["mean"].detach().cpu().numpy())
            if not only_mean:
                counts_impute_disp.append(expr_pred["disp"].detach().cpu().numpy())
            
            if "label" in data_sample.keys():
                labels.append(data_sample["label"].reshape(-1).detach().cpu().numpy())
            else:
                labels.append(np.array([np.nan] * cell_embed.shape[0]))

            if "batch" in data_sample.keys():
                batches.append(data_sample["batch"].reshape(-1).detach().cpu().numpy())
            else:
                batches.append(np.array([np.nan] * cell_embed.shape[0]))

    # usually dense
    counts_impute_norm = np.vstack(counts_impute_norm)
    labels = np.concatenate(labels, axis = 0)
    batches = np.concatenate(batches, axis = 0)
    meta = pd.DataFrame.from_dict({"label_id": labels, "batch_id": batches})
    adata = anndata.AnnData(X = counts_impute_norm, obs = meta.astype("category"))
    adata.layers["mean"] = counts_impute_norm

    if not only_mean:
        counts_impute_disp = np.vstack(counts_impute_disp)
        adata.layers["disp"] = counts_impute_disp

    return adata




def train_multigpus(model, train_config, optimizer, scheduler, writer):

    """
    Description:
    ------------
        The training function of foundation model

    Parameters:
    ------------
        model: transformer model
        train_config: dict of training setting
        optimizer: the optimizer of the model
        scheduler: the scheduler of the model
        writer: logging

    """
    global_rank = train_config["global_rank"]
    log_step = train_config["log_step"]
    initial_epoch = train_config["initial_epoch"]
    initial_step = train_config["initial_step"]

    print(f"GPU {global_rank} - Loading dataset...")
    # Need to normalize the data, the min_chunksize = 64, so batchsize 512 = 8 samples * 64
    min_chunksize = 64
    label_colname = train_config["label_colname"]
    batch_colname = train_config["batch_colname"]
    val_dataset = data_utils.sc_partition(data_path = train_config["DIR"], batch_feats = train_config["batch_dict"], min_chunksize = min_chunksize, normalize = model.module.model_config.lognorm_data)
    val_dataset.load_partition(idx = train_config["num_partitions"] - 1, label_colname = label_colname, batch_colname = batch_colname, data_prefix = train_config["data_prefix"], meta_prefix = train_config["meta_prefix"]) # use last chunk
    val_loader = data.DataLoader(val_dataset, batch_size = model.module.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True,
                                 sampler = DistributedSampler(val_dataset, shuffle = False), num_workers = 8, prefetch_factor = 8)
    train_dataset = data_utils.sc_partition(data_path = train_config["DIR"], batch_feats = train_config["batch_dict"], min_chunksize = min_chunksize, normalize = model.module.model_config.lognorm_data)
    print(f"GPU {global_rank} - Done.")

    scaler = GradScaler()
    # NOTE: training loop
    checkpoint_counter = 0
    if model.module.model_config.use_flashatten:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    for epoch in range(initial_epoch, model.module.model_config.n_epoch):
        step = 0
        running_loss, running_loss_mlm, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0

        # shuffle the partition for each epoch
        for partition_idx in np.random.permutation(train_config["num_partitions"] - 1):
            print(f"GPU {global_rank} - Start training Epoch {epoch:02d}, Partition {partition_idx:02d}...")
            torch.cuda.empty_cache()
            # load training dataset for partition_idx
            train_dataset.load_partition(idx = partition_idx, label_colname = label_colname, batch_colname = batch_colname, data_prefix = train_config["data_prefix"], meta_prefix = train_config["meta_prefix"])
            # shuffle in distributed sampler
            train_loader = data.DataLoader(train_dataset, batch_size = model.module.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True,
                                        sampler = DistributedSampler(train_dataset, shuffle = True), num_workers = 8, prefetch_factor = 8)

            batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}, Partition {partition_idx:02d} on rank {global_rank}", disable = global_rank != 0)

            for data_sample in batch_iterator:
                model.train()
                optimizer.zero_grad()
            
                if step < initial_step:
                    step += 1
                    continue

                with autocast(device_type="cuda", dtype = CAST_DTYPE):
                    loss, loss_item = infer_databatch(model, data_sample, multigpus = True)

                scaler.scale(loss).backward()

                #Unscale the optimizer to clip gradients
                scaler.unscale_(optimizer)
                # clip gradient
                max_grad_norm = 1.0 
                clip_grad_norm_(model.parameters(), max_grad_norm)


                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # NOTE: log the results
                running_loss += loss.item()
                running_loss_mlm += loss_item["mlm"]
                running_loss_metric += loss_item["metric"]
                running_loss_mincut += loss_item["mincut"]
                running_loss_ortho += loss_item["ortho"]
            
                if step % log_step == log_step - 1:
                    # calculate for each gpus
                    running_loss /= log_step
                    running_loss_mlm /= log_step
                    running_loss_metric /= log_step
                    running_loss_mincut /= log_step
                    running_loss_ortho /= log_step
                    if writer is not None:
                        cum_step = epoch * len(train_loader) * train_config["num_partitions"] + step + 1
                        # only write/print the running loss for one gpu with writer
                        writer.add_scalar("Train Loss (TOTAL)", running_loss, cum_step)
                        writer.add_scalar("Train Loss (MLM)", running_loss_mlm, cum_step)
                        writer.add_scalar("Train Loss (METRIC)", running_loss_metric, cum_step)
                        writer.add_scalar("Train Loss (MINCUT)", running_loss_mincut, cum_step)
                        writer.add_scalar("Train Loss (ORTHO)", running_loss_ortho, cum_step)
                        writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], cum_step)
                        print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * train_config["num_partitions"]}, Learning rate: {scheduler.get_last_lr()[0]:.2e}, Mask prob: {model.module.model_config.mask_prob:.4f}, \
                              Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM):{running_loss_mlm:.4f}, Train Loss (METRIC): {running_loss_metric:.4f}, Train Loss (MINCUT): {running_loss_mincut:.4f}, Train Loss (ORTHO): {running_loss_ortho:.4f}")

                    running_loss, running_loss_mlm, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0
                    checkpoint_counter += 1

                    # model evaluation and checkpoint saving
                    if (checkpoint_counter == 10):
                        model.eval()
                        with torch.no_grad():
                            val_loss, val_loss_mlm, val_loss_metric, val_loss_mincut, val_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0
                            for data_sample in val_loader:
                                with autocast(device_type="cuda", dtype = CAST_DTYPE):

                                    loss, loss_item = infer_databatch(model, data_sample, multigpus = True)                            
                                    val_loss += loss.item()
                                    val_loss_mlm += loss_item["mlm"]
                                    val_loss_metric += loss_item["metric"]
                                    val_loss_mincut += loss_item["mincut"]
                                    val_loss_ortho += loss_item["ortho"]

                            # log the values
                            val_loss /= len(val_loader)
                            val_loss_mlm /= len(val_loader)
                            val_loss_metric /= len(val_loader)
                            val_loss_mincut /= len(val_loader)
                            val_loss_ortho /= len(val_loader)
                            
                            if writer is not None:
                                cum_step = epoch * len(train_loader) * train_config["num_partitions"] + step + 1
                                writer.add_scalar("Val Loss (TOTAL)", val_loss, cum_step)
                                writer.add_scalar("Val Loss (MLM)", val_loss_mlm, cum_step)
                                writer.add_scalar("Val Loss (METRIC)", val_loss_metric, cum_step)
                                writer.add_scalar("Val Loss (MINCUT)", val_loss_mincut, cum_step)
                                writer.add_scalar("Val Loss (ORTHO)", val_loss_ortho, cum_step)
                                writer.add_scalar("Mask prob", model.module.model_config.mask_prob, cum_step)
                                print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * train_config["num_partitions"]}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, \
                                      Val Loss (METRIC): {val_loss_metric:.4f}, Val Loss (MINCUT): {val_loss_mincut:.4f}, Val Loss (ORTHO): {val_loss_ortho:.4f}")

                                # save only for the writer gpu
                                save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                                                path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{epoch}_{step + 1}.pth")
                        
                        checkpoint_counter = 0                

                    # sync all gpus after eval
                    dist.barrier()

                # update the freezing status, after finish the first epoch
                if (model.module.model_config.mlm_type != "meta") and (step == int(0.1 * len(train_loader) * train_config["num_partitions"])):
                    print(f"unfreeze the model at step {step:d}")
                    model.module.freeze_fm_gradient(freeze_trans = False, freeze_batchenc = False, freeze_compression = False)

                # update the mask_prob 
                if model.module.model_config.dynamic_maskprob:
                    model.module.mask_scheduler.step()
                    model.module.model_config.mask_prob = model.module.mask_scheduler.get_prob()
                step += 1   
        # end of the epoch, reset the init step as 0
        initial_step = 0

    # save the final model, also only for the writer gpu
    if writer is not None:
        save_checkpoint(epoch = model.module.model_config.n_epoch, step = 0, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                        path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{model.module.model_config.n_epoch}.pth")
        




def train_singlegpu(model, train_config, optimizer, scheduler, writer):

    """
    Description:
    ------------
        The training function of foundation model

    Parameters:
    ------------
        model: transformer model
        train_loader: the training data loader
        val_loader: the validation data loader
        optimizer: the optimizer of the model
        scheduler: the scheduler of the model
        writer: the tensorboard writer
        TODO: ADD
    """

    log_step = train_config["log_step"]
    initial_epoch = train_config["initial_epoch"]
    initial_step = train_config["initial_step"]

    print(f"GPU - Loading dataset...")
    # Need to normalize the data, the min_chunksize = 64, so batchsize 512 = 8 samples * 64
    min_chunksize = 64
    label_colname = train_config["label_colname"]
    batch_colname = train_config["batch_colname"]
    val_dataset = data_utils.sc_partition(data_path = train_config["DIR"], batch_feats = train_config["batch_dict"], min_chunksize = min_chunksize, normalize = model.model_config.lognorm_data)
    val_dataset.load_partition(idx = train_config["num_partitions"] - 1, label_colname = label_colname, batch_colname = batch_colname, data_prefix = train_config["data_prefix"], meta_prefix = train_config["meta_prefix"]) # use last chunk
    val_loader = data.DataLoader(val_dataset, batch_size = model.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    train_dataset = data_utils.sc_partition(data_path = train_config["DIR"], batch_feats = train_config["batch_dict"], min_chunksize = min_chunksize, normalize = model.model_config.lognorm_data)
    print(f"GPU - Done.")

    scaler = GradScaler()
    # NOTE: training loop
    checkpoint_counter = 0
    if model.model_config.use_flashatten:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    for epoch in range(initial_epoch, model.model_config.n_epoch):
        step = 0
        running_loss, running_loss_mlm, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0

        # shuffle the partition for each epoch
        for partition_idx in np.random.permutation(train_config["num_partitions"] - 1):
            print(f"GPU - Start training Epoch {epoch:02d}, Partition {partition_idx:02d}...")
            torch.cuda.empty_cache()
            # load training dataset for partition_idx
            train_dataset.load_partition(idx = partition_idx, label_colname = label_colname, batch_colname = batch_colname, data_prefix = train_config["data_prefix"], meta_prefix = train_config["meta_prefix"])
            # shuffle in distributed sampler
            train_loader = data.DataLoader(train_dataset, batch_size = model.model_config.batch_size//min_chunksize, shuffle = True, pin_memory = True, num_workers = 8, prefetch_factor = 8)

            batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}, Partition {partition_idx:02d}")

            for data_sample in batch_iterator:
                model.train()
                optimizer.zero_grad()
            
                if step < initial_step:
                    step += 1
                    continue

                with autocast(device_type="cuda", dtype = CAST_DTYPE):
                    loss, loss_item = infer_databatch(model, data_sample, multigpus = False)

                scaler.scale(loss).backward()

                #Unscale the optimizer to clip gradients
                scaler.unscale_(optimizer)
                # clip gradient
                max_grad_norm = 1.0 
                clip_grad_norm_(model.parameters(), max_grad_norm)


                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # NOTE: log the results
                running_loss += loss.item()
                running_loss_mlm += loss_item["mlm"]
                running_loss_metric += loss_item["metric"]
                running_loss_mincut += loss_item["mincut"]
                running_loss_ortho += loss_item["ortho"]

                if step % log_step == log_step - 1:
                    # calculate for each gpus
                    running_loss /= log_step
                    running_loss_mlm /= log_step
                    running_loss_metric /= log_step
                    running_loss_mincut /= log_step
                    running_loss_ortho /= log_step

                    if writer is not None:
                        cum_step = epoch * len(train_loader) * train_config["num_partitions"] + step + 1
                        # only write/print the running loss for one gpu with writer
                        writer.add_scalar("Train Loss (TOTAL)", running_loss, cum_step)
                        writer.add_scalar("Train Loss (MLM)", running_loss_mlm, cum_step)
                        writer.add_scalar("Train Loss (METRIC)", running_loss_metric, cum_step)
                        writer.add_scalar("Train Loss (MINCUT)", running_loss_mincut, cum_step)
                        writer.add_scalar("Train Loss (ORTHO)", running_loss_ortho, cum_step)
                        writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], cum_step)

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * train_config["num_partitions"]}, Learning rate: {scheduler.get_last_lr()[0]:.2e}, Mask prob: {model.model_config.mask_prob:.4f}, \
                            Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM):{running_loss_mlm:.4f}, Train Loss (METRIC): {running_loss_metric:.4f}, Train Loss (MINCUT): {running_loss_mincut:.4f}, Train Loss (ORTHO): {running_loss_ortho:.4f}")
                    
                    running_loss, running_loss_mlm, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0
                    checkpoint_counter += 1

                    # model evaluation and checkpoint saving
                    if (checkpoint_counter == 50):
                        model.eval()
                        with torch.no_grad():
                            val_loss, val_loss_mlm, val_loss_metric, val_loss_mincut, val_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0
                            for data_sample in val_loader:
                                with autocast(device_type="cuda", dtype = CAST_DTYPE):

                                    loss, loss_item = infer_databatch(model, data_sample, multigpus = False)                            
                                    val_loss += loss.item()
                                    val_loss_mlm += loss_item["mlm"]
                                    val_loss_metric += loss_item["metric"]
                                    val_loss_mincut += loss_item["mincut"]
                                    val_loss_ortho += loss_item["ortho"]

                            # log the values
                            val_loss /= len(val_loader)
                            val_loss_mlm /= len(val_loader)
                            val_loss_metric /= len(val_loader)
                            val_loss_mincut /= len(val_loader)
                            val_loss_ortho /= len(val_loader)

                            if writer is not None:
                                cum_step = epoch * len(train_loader) * train_config["num_partitions"] + step + 1
                                writer.add_scalar("Val Loss (TOTAL)", val_loss, cum_step)
                                writer.add_scalar("Val Loss (MLM)", val_loss_mlm, cum_step)
                                writer.add_scalar("Val Loss (METRIC)", val_loss_metric, cum_step)
                                writer.add_scalar("Val Loss (MINCUT)", val_loss_mincut, cum_step)
                                writer.add_scalar("Val Loss (ORTHO)", val_loss_ortho, cum_step)
                                writer.add_scalar("Mask prob", model.model_config.mask_prob, cum_step)

                            print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * train_config["num_partitions"]}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, \
                                    Val Loss (METRIC): {val_loss_metric:.4f}, Val Loss (MINCUT): {val_loss_mincut:.4f}, Val Loss (ORTHO): {val_loss_ortho:.4f}")

                            # save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                            #                 path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{epoch}_{step + 1}.pth", multi_gpus = False)
                                                
                        checkpoint_counter = 0                


                # update the freezing status, after finish the first epoch
                if (model.model_config.mlm_type != "meta") and (step == int(0.1 * len(train_loader) * train_config["num_partitions"])):
                    print(f"unfreeze the model at step {step:d}")
                    model.freeze_fm_gradient(freeze_trans = False, freeze_batchenc = False, freeze_compression = False)

                # update the mask_prob 
                if model.model_config.dynamic_maskprob:
                    model.mask_scheduler.step()
                    model.model_config.mask_prob = model.mask_scheduler.get_prob()
                    
                step += 1   
        initial_step = 0

    save_checkpoint(epoch = model.model_config.n_epoch, step = 0, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                    path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{model.model_config.n_epoch}.pth", multi_gpus = False)
