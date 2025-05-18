import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func

    flash_attn_available = True
except ImportError:
    raise ValueError("flash attention is not correctly installed")


class FlashTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = "relu"):
        super(FlashTransformerLayer, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Q, K, V projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation can only be 'relu' or 'gelu'")

    def forward(self, x, src_key_padding_mask=None):
        # x shape: (seq_len, batch, d_model) -> transpose to (batch, seq_len, d_model)
        x = x.transpose(0, 1)
        batch, seq_len, _ = x.shape

        if src_key_padding_mask is not None:
            raise NotImplementedError("FlashAttention V2 does not support key padding mask.")

        # Compute Q, K, V projections. shape (batch, seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, nhead, head_dim)
        q = q.view(batch, seq_len, self.nhead, self.head_dim)
        k = k.view(batch, seq_len, self.nhead, self.head_dim)
        v = v.view(batch, seq_len, self.nhead, self.head_dim)

        # Stack Q, K, V into one tensor of shape (batch, seq_len, 3, nhead, head_dim)
        qkv = torch.stack((q, k, v), dim=2)

        softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # The output will have shape (batch, seq_len, nhead, head_dim)
        attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout1.p, softmax_scale=softmax_scale, causal=False)

        # Reshape to (batch, seq_len, d_model) and project to output dimension
        attn_output = attn_output.reshape(batch, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_output))

        # Transpose back to original shape: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        return x


class TransformerBlocks(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_layers: int, dim_feedforward: int, dropout: float, activation: str = "gelu"):
        super(TransformerBlocks, self).__init__()
        self.layers = nn.ModuleList([
            FlashTransformerLayer(d_model, n_head, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src

