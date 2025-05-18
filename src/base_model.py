from typing import Iterable
import torch
from torch import nn as nn
import torch.nn.functional as F
import collections
from torch.amp import autocast

from torch.autograd import Function
# gradient reversal, necessary for discriminator training
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def gradient_reversal(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class MinMaxNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(MinMaxNormalization, self).__init__()
        self.eps = eps  # A small value to avoid division by zero

    def forward(self, x):
        # Compute the min and max along the specified dimensions
        x_max = x.max(dim=1, keepdim=True)[0]
        # x_min is always 0
        
        # Normalize to [0, 1]
        x_normalized = x / (x_max + self.eps)
        return x_normalized

class LogNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6, scale_factor: float = 10e4):
        super(LogNormalization, self).__init__()
        self.eps = eps
        self.scale_factor = scale_factor
    def forward(self, x, norm_1: bool):
        x = x/(x.sum(dim = 1, keepdim = True) + self.eps) * self.scale_factor
        x = torch.log1p(x)
        if norm_1:
            x = x/(x.sum(1, keepdim = True) + self.eps)
        return x
    
    
def identity(x):
    return x

def one_hot(index, n_cat, dtype = torch.bfloat16) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(dtype)

def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )

def fourier_positional_encoding(x: torch.Tensor, embedding_dim: int):
    """
    Computes the Fourier-based positional embedding for a continuous value x in [0, 1].
    
    Args:
        x (float): A continuous value in the range [0, 1].
        embedding_dim (int): The dimensionality of the positional embedding.
        
    Returns:
        numpy.ndarray: The positional embedding vector.
    """
    # assert torch.max(x) <= 1, "x must be between 0 and 1"
    # assert torch.min(x) >= 0
    # Half of the embedding dimension will be used for sin, and half for cos
    half_dim = embedding_dim // 2
    
    # Define the frequencies as powers of 2, add 0.1 scaling to shrink value range (not necessary)
    frequencies = 0.1 * 2 ** torch.arange(half_dim).to(x.device)
    
    # Compute the sine and cosine components
    sin_components = torch.sin(frequencies * x)
    cos_components = torch.cos(frequencies * x)
    
    # Concatenate the sin and cos components
    positional_embedding = torch.concat([sin_components, cos_components], dim = -1)
    
    return positional_embedding

def log_nb_positive(x, mu, theta, eps=1e-8):
    """
    Note: All inputs should be torch Tensors
    log likelihood (scalar) of a minibatch according to a nb model.

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = theta * (torch.log(theta + eps) - log_theta_mu_eps) + \
        x * (torch.log(mu + eps) - log_theta_mu_eps) + \
        torch.lgamma(x + theta) - \
        torch.lgamma(theta) - \
        torch.lgamma(x + 1)

    return res # torch.sum(res, dim=-1)

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = "relu"):
        super(TransformerLayer, self).__init__()
        
        # Multi-head self attention
        self.self_attn = nn.MultiheadAttention(embed_dim = d_model, num_heads = nhead, dropout = dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation can only be `relu' or `gelu'")


    def forward(self, x, src_key_padding_mask = None):
        # Apply mixed precision to the entire TransformerEncoderLayer
        attn_output = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)[0]
        
        # Force layer normalization and softmax to run in FP32 for stability
        with autocast(device_type = "cuda", enabled=False):
            x = self.norm1(x + self.dropout1(attn_output))
        
        # Continue the feedforward part in mixed precision
        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # Again, force normalization to run in FP32
        with autocast(device_type = "cuda", enabled=False):
            x = self.norm2(x + self.dropout2(feedforward_output))
        
        return x
    

class TransformerBlocks(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_layers: int, dim_feedforward: int, dropout: float, activation: str = "gelu"):
        super(TransformerBlocks, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_head, dim_feedforward, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, src, src_key_padding_mask = None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src 





# NOTE: batch-effect related factors include 
# 1. the continuous variables: library size, non-zero gene expression level, house keeping genes, etc
# 2. the categorical variables: sequencing technologies, donor conditions, etc
# NOTE: issue, is the categorical values independent or related (independent use nn.Embed), most factors have independent categorical effect on batch effect

class batch_encoder_mlp(nn.Module):
    def __init__(self, n_feat, n_embed, p_drop = 0.1):
        super().__init__()
        self.n_feat = n_feat
        self.n_embed = n_embed
        self.enc_cont = nn.Sequential(full_block(in_features = self.n_feat, out_features = self.n_embed, p_drop = p_drop), 
                                      full_block(in_features = self.n_embed, out_features = self.n_embed, p_drop = p_drop))    
    
    def forward(self, batch_factors):
        batch_embed = self.enc_cont(batch_factors)
        weight = torch.ones((self.n_feat,)).to(batch_factors.device)
        return batch_embed, weight


class batch_encoder_cont(nn.Module):
    def __init__(self, n_feat, n_embed, p_drop = 0.1):
        super().__init__()
        self.n_feat = n_feat
        self.n_latent = 32
        self.n_embed = n_embed
        for id in range(self.n_feat):
            self.enc_cont[str(id)] = nn.Sequential(full_block(in_features = self.n_latent, out_features = self.n_latent, p_drop = p_drop), 
                                                   nn.Linear(in_features = self.n_latent, out_features = self.n_embed))    

        # parameter weight
        self.weight = nn.Parameter(torch.randn((self.n_feat)))
        self.softmax = nn.Softmax()
    
    def forward(self, batch_factors):
        weight = self.softmax(self.weight)
        batch_embed = torch.zeros((batch_factors.shape[0], self.n_embed), device = batch_factors.device)

        for id in range(self.n_feat):
            batch_embed += weight[id] * self.enc_cont[str(id)](fourier_positional_encoding(batch_factors[:, [id]], embedding_dim = self.n_latent))
        return batch_embed, weight


class batch_encoder_cat(nn.Module):
    def __init__(self, n_cat_list, n_embed, n_output, p_drop):
        super().__init__()
        self.n_cat_feat = len(n_cat_list)
        self.n_embed = n_embed
        self.n_output = n_output
        self.p_drop = p_drop

        self.enc_cat = nn.ModuleDict({})
        for id in range(self.n_cat_feat):
            self.enc_cat[str(id)] = nn.Embedding(num_embeddings = n_cat_list[id], embedding_dim = self.n_embed)   
             
        self.enc_proj = full_block(self.n_cat_feat * self.n_embed, self.n_output, p_drop = self.p_drop)

    def forward(self, batch_factors):
        batch_embed = []
        for id in range(self.n_cat_feat):
            x_cat = batch_factors[:, [id]].long()
            embed = self.enc_cat[str(id)](x_cat).squeeze()
            batch_embed.append(embed) 
        batch_embed = torch.hstack(batch_embed)
        # batch_embed = nn.functional.normalize(self.enc_proj(batch_embed))
        batch_embed = self.enc_proj(batch_embed)

        weights = torch.ones((self.n_cat_feat,)).to(batch_embed.device)
        return batch_embed, weights


# basically copying the encoder_batchfactor and remove redundancy
class batch_encoder_catpool(nn.Module):
    def __init__(self, n_cat_list, n_embed):
        super().__init__()
        self.n_cat_feat = len(n_cat_list)
        self.n_embed = n_embed

        self.enc_cat = nn.ModuleDict({})
        for id in range(self.n_cat_feat):
            self.enc_cat[str(id)] = nn.Embedding(num_embeddings = n_cat_list[id], embedding_dim = self.n_embed)            
    
        self.enc_proj = full_block(self.n_cat_feat * self.n_embed, self.n_output, p_drop = self.p_drop)

        # parameter weight
        self.weight = nn.Parameter(torch.randn((self.n_cat_feat)))
        self.softmax = nn.Softmax()
    
    def forward(self, batch_factors):
        weight = self.softmax(self.weight)
        batch_embed = torch.zeros((batch_factors.shape[0], self.n_embed), device = batch_factors.device)

        for id in range(self.n_cat_feat):
            x_cat = batch_factors[:, [id]].long()
            # # with 0 assignment for -1
            # mask = (x_cat == -1).squeeze()
            # x_cat[mask] = 0
            embed = self.enc_cat[str(id)](x_cat).squeeze()
            # embed[mask] = 0.0  
            batch_embed += weight[id] * embed

        batch_embed = self.enc_proj(batch_embed)

        return batch_embed, weight

