import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base_model

# ----------------------------------------------------------------------------------------
#
# Learnable gene grouping
#
# ----------------------------------------------------------------------------------------
def knn_graph(features, k, include_self: bool = True):
    """
    features: Tensor of shape [N, D] (N nodes, D features)
    k: Number of nearest neighbors
    Returns: Adjacency matrix [N, N] (binary, symmetric)
    """
    print("construct knn graph from embedding.")
    N = features.size(0)

    # Compute pairwise squared Euclidean distance matrix: [N, N]
    dist = torch.cdist(features, features, p=2)  # L2 distance

    # Mask self-distances
    dist.fill_diagonal_(float('inf'))

    # Find k smallest distances per row (i.e. nearest neighbors)
    knn_indices = dist.topk(k, dim=1, largest=False).indices  # [N, k]

    # Create adjacency matrix
    A = torch.zeros(N, N, device=features.device)
    A.scatter_(1, knn_indices, 1.0)  # Directed edges

    # Make the graph undirected (optional)
    A = torch.maximum(A, A.T)
    A = A + torch.eye(A.size(0)).to(A.device)

    return A


def mincut_loss(A, S, add_orthogonality=True, eps=1e-9):
    """
    A: [N, N] Adjacency matrix (un-normalized)
    S: [N, k] Soft pooling assignment matrix (softmax over k)
    """
    N, k = S.shape
    D = torch.diag(A.sum(dim=1))  # Degree matrix

    # MinCut numerator and denominator
    cut = torch.trace(S.T @ A @ S)
    assoc = torch.trace(S.T @ D @ S)
    mincut_loss = -cut / (assoc + eps)

    if add_orthogonality:
        # Orthogonality regularization
        # SS = S.T @ S
        # SS_norm = SS / (torch.norm(S, p='fro')**2 + eps)

        S_norm = S/S.pow(2).sum(0, keepdim = True).sqrt()
        SS_norm = S_norm.T @ S_norm
        I = torch.eye(k, device=A.device)
        orth_loss = torch.norm(SS_norm - I, p='fro')

    else:
        orth_loss = torch.tensor([0]).to(mincut_loss.device)

    return mincut_loss, orth_loss
    

class GenePoolVanilla(nn.Module):
    def __init__(self, A, n_meta = 256, s_init = None, temp = 0.5):
        super(GenePoolVanilla, self).__init__()
        # Naive gene pool, no random walk, directly parameterize the pooling score
        self.A = A
        if s_init is None:
            self.S = nn.Parameter(torch.rand((self.A.shape[0], n_meta)))
        else:
            self.S = nn.Parameter(s_init.clone())
        self.softmax = nn.Softmax(dim = 1)
        self.temp = temp
        # self.normalization = base_model.MinMaxNormalization(eps = 1e-6)

    def get_score(self):
        return self.softmax(self.S/self.temp)
    
    def forward(self, expr, gene_embed, log_norm: bool = True):
        S = self.get_score()
        expr_pool = expr @ S
        gene_pool = S.T @ gene_embed

        if log_norm:
            expr_pool = expr_pool/(torch.sum(expr_pool, dim = 1, keepdim = True) + 1e-4) * 10e4
            expr_pool = torch.log1p(expr_pool)
        
        return S, gene_pool, expr_pool # self.normalization(expr_pool)
    

    def mincut_loss(self, add_ortho: bool = False):
        return mincut_loss(self.A, self.get_score(), add_orthogonality = add_ortho)
