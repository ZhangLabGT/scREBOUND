import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.functional import sigmoid, softmax

import torch.distributed.nn.functional as dist_func

def compute_cross_entropy(p, q):
    q = nn.functional.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    NOTE: requires that the tensor to be the same size across gpus, which is not true with the filtered samples
    """
    # dist.get_world_size() returns the number of gpus
    # create a list of n all-one tensor of size the same as input tensor, n is number of gpu
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    # gather input tensor from all gpus, and fill in the tensors_gather
    # NOTE: pass only tensor not gradient
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    # output = tensors_gather
    return output


def concat_all_gather_gradient(tensor):
    """
    Performs all_gather operation on the provided tensors. Slower
    """
    tensors_gather = dist_func.all_gather(tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output

def exact_equal(A, B, chunk_size = 16):
    # result = torch.all(torch.eq(A[:, None, :], B[None, :, :]), dim=2) 
    # return result

    results = []
    for start in range(0, A.shape[0], chunk_size):
        end = start + chunk_size
        partial = torch.all(A[start:end, None, :] == B[None, :, :], dim=2)
        results.append(partial)
    result = torch.cat(results, dim=0)
    return result

def full_equal(A, B):
    return (A @ B.T) != 0


def find_ancestor(A, B, chunk_size = 16):
    results = []
    for start in range(0, A.shape[0], chunk_size):
        end = start + chunk_size
        # partial = torch.sum(A[start:end, None, :].bool() | B[None, :, :].bool(), dim=2)
        partial = torch.sum(torch.logical_or(A[start:end, None, :], B[None, :, :]), dim=2)
        results.append(partial)

    result = torch.cat(results, dim=0)
    return result <= torch.sum(A, dim = 1, keepdim = True)

def find_descend(A, B):
    """ 
    Find the descendent cells
    using bincode, the descendent cells should have 1 for all 1s in the cell bincode
    including itself
    return a cell by cell binary matrix, where 1 denote descendent, 0 denote non-descendent
    """
    result = A @ B.T 
    return result == torch.sum(A, dim = 1, keepdim = True)


class SupContrLoss(nn.Module):
    """
    Supervised contrastive loss
    """
    def __init__(self, label_asso_mtx: torch.Tensor, temperature: float = 0.1):
        super(SupContrLoss, self).__init__()
        self.label_asso_mtx = label_asso_mtx
        self.temperature = temperature
    
    def forward(self, features, label_ids, batch_ids = None):
        """
        features: feature embedding, of shape (ncells, nfeats)
        """
        device = features.device
        
        # calculate the label association matrix given the samples
        label_asso = self.label_asso_mtx[label_ids.unsqueeze(1), label_ids.unsqueeze(0)]

        # update the label_asso is batch is not None
        if batch_ids is not None:
            # NOTE: if the batch label is provided, the contrastive loss is applied only across batches to better remove batch effect
            # positive sample only includes the samples of the same cell type across batchs, but should the samples of the same cell type within the same batch be negative samples?
            batch_ids = batch_ids.contiguous().view(-1, 1)
            # (ncells, ncells) batch identification matrix, sample cell type in same batch: 1, remaining batch: 0
            extra_neutral = torch.eq(batch_ids, batch_ids.T).float().to(device) * (label_asso == 1).float()
            # remove self-similarity
            assert torch.all(torch.diag(extra_neutral) == 1)
            # these samples are neutral
            label_asso[extra_neutral.bool()] = 0
        else:
            # remove self-similarity
            extra_neutral = torch.eye(label_asso.shape[0]).to(label_asso.device)
            label_asso[extra_neutral] = 0

        # -------------------------------------------
        # Contrastive loss with ground truth matrix provided, label_asso
        # compute logits
        logits = torch.matmul(features, features.T) / self.temperature
        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        neutral_label = (label_asso == 0)
        logits.masked_fill_(neutral_label, -1e7)
        # compute ground-truth distribution
        # for each sample, sum mask between the sample and all remaining samples, and clamp by 1 min to prevent 0 sum
        # more neighboring, more even the p is after normalization
        pos_label = (label_asso == 1).float()
        p = pos_label / pos_label.sum(1, keepdim=True).clamp(min=1.0)

        # cross entropy loss, select and calculate only on non-neutral mask 
        loss = compute_cross_entropy(p, logits)

        # -------------------------------------------

        return loss
    


class SupContrLossMultiGPUs(nn.Module):
    """
    Supervised contrastive loss
    """
    def __init__(self, label_asso_mtx: torch.Tensor, temperature: float = 0.1, unknown_label: int | None = None):
        super(SupContrLossMultiGPUs, self).__init__()
        self.label_asso_mtx = label_asso_mtx
        self.unknown_label = unknown_label
        self.temperature = temperature
    
    def forward(self, features, label_ids, batch_ids = None):
        """
        features: feature embedding, of shape (ncells, nfeats)
        label_asso: binary identification matrix, of shape (ncells, ncells), pos: 1, neutral: 0, neg: -1
        """
        device = features.device

        local_batch_size = features.size(0)
        # get all features across gpus
        all_features = concat_all_gather_gradient(features)
        # get all labels across gpus
        all_label_ids = concat_all_gather(label_ids)

        # label by all_labels association matrix
        label_asso = self.label_asso_mtx[label_ids.unsqueeze(1), all_label_ids.unsqueeze(0)]

        # remove self-similarity, 0 for self-similarity
        self_sim_mask = torch.scatter(torch.ones_like(label_asso), 1,
                                      torch.arange(label_asso.shape[0]).view(-1, 1).to(device) +
                                      local_batch_size * dist.get_rank(),
                                      0).bool().to(device)

        # update the label_asso is batch is not None
        if batch_ids is not None:
            # NOTE: if the batch label is provided, the contrastive loss is applied only across batches to better remove batch effect
            # positive sample only includes the samples of the same cell type across batchs, but should the samples of the same cell type within the same batch be negative samples?
            # cheaper than reshape(-1, 1)
            batch_ids = batch_ids.contiguous().view(-1, 1)
            all_batch_ids = concat_all_gather(batch_ids)

            # (ncells, ncells) batch identification matrix, sample cell type in same batch: 1, remaining batch: 0
            extra_neutral = torch.eq(batch_ids, all_batch_ids.T).float().to(device) * (label_asso == 1).float()
            # remove self-similarity
            assert torch.all(extra_neutral[~self_sim_mask] == 1)
            # these samples are neutral
            label_asso[extra_neutral.bool()] = 0
        else:
            # remove self-similarity
            label_asso[~self_sim_mask] = 0


        # -------------------------------------------
        # Contrastive loss with ground truth matrix provided, label_asso
        # compute logits
        logits = torch.matmul(features, all_features.T) / self.temperature

        # drop the unknown after calculating all the metrics
        if self.unknown_label is not None:
            keep_idx = (label_ids != self.unknown_label)
            all_keep_idx = (all_label_ids != self.unknown_label)
            
            label_asso = label_asso[keep_idx][:, all_keep_idx]
            logits = logits[keep_idx][:, all_keep_idx]

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)
        neutral_label = (label_asso == 0)
        logits.masked_fill_(neutral_label, -1e7)
        # compute ground-truth distribution
        # for each sample, sum mask between the sample and all remaining samples, and clamp by 1 min to prevent 0 sum
        # more neighboring, more even the p is after normalization
        pos_label = (label_asso == 1).float()
        p = pos_label / pos_label.sum(1, keepdim=True).clamp(min=1.0)

        # cross entropy loss, select and calculate only on non-neutral mask 
        loss = compute_cross_entropy(p, logits)
        # -------------------------------------------

        return loss
