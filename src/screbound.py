
# Directly use the transformer provided by torch
import torch
import torch.nn as nn
import math
import numpy as np

from dataclasses import dataclass
import base_model
from data_utils import set_seed
import graph_pool
import contrastive

try:
    import flash_transformer_layer as flash_model
    use_flashatten = True
except:
    use_flashatten = False

@dataclass
class ModelConfig:

    batch_size: int # Batch size
    n_epoch: int # Number of epochs to train
    lr: float # Learning rate
    d_embed: int
    n_head: int
    d_hidden: int
    n_layer: int
    d_output: int
    dropout: float

    # training
    mask_prob: float
    dynamic_maskprob: bool

    # compression
    lamb_mincut: float
    mlm_type: str
    lamb_mlm: float

    # batch & contr reg
    batch_enc: dict | None
    use_contr: bool
    lamb_contr: float

    # model stats
    precision: torch.dtype

    # paths
    checkpoint_path: str
    checkpoint_prefix: str
    pretrain_path: str | None

    lognorm_data: bool

def get_default_config() -> ModelConfig:

    return ModelConfig(
        batch_size=1024,
        n_epoch=1,
        lr=5e-4,
        d_embed=512, # dimension of each head == d_embed/n_head 512 = 8*64, 768 = 12*64
        n_head=8,
        d_hidden=2048,
        n_layer=4,
        d_output=256,
        dropout=0.1,
        mask_prob=0.4,
        dynamic_maskprob=False,
        lamb_mincut=1,
        mlm_type="meta",
        lamb_mlm=1,

        batch_enc={"encoder": "encoder", "n_cat_list": None},
        use_contr=False,
        lamb_contr=0,
        
        precision=torch.float32,
        pretrain_path=None,
        checkpoint_path="checkpoint/",
        checkpoint_prefix="checkpoint",
        lognorm_data=True
    )


class ConditionalPredictor(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, output_dim: int, n_layers: int, dropout: float, deep_inj: bool, nb_head: bool = False):
        super().__init__()

        latent_dim = 512
        self.deep_inj = deep_inj
        self.nb_head = nb_head
        self.layers = nn.ModuleList()
        self.layers.append(base_model.full_block(input_dim + condition_dim, latent_dim, p_drop = dropout))

        for layer in range(1, n_layers - 1):
            self.layers.append(base_model.full_block(latent_dim + condition_dim if self.deep_inj else latent_dim, latent_dim, p_drop = dropout))

        if not self.nb_head:
            self.layers.append(nn.Linear(latent_dim + condition_dim if self.deep_inj else latent_dim, output_dim))
            # predict non-negative values
            # self.activation = nn.Sigmoid()
            self.activation = nn.Softplus()

        else:
            self.mean_head = nn.Sequential(nn.Linear(latent_dim, output_dim),
                                           nn.Softmax(dim = -1))
            
            self.dispersion_head = nn.Sequential(nn.Linear(latent_dim, output_dim))

    def forward(self, x, cond):
        for idx, layer in enumerate(self.layers):
            if (not self.deep_inj) and (idx > 0):
                x = layer(x)
            else:
                if cond is not None:
                    x = layer(torch.cat([x, cond], dim = 1))
                else:
                    x = layer(x)

        if not self.nb_head:
            x = self.activation(x)
            return x
        
        else:
            x_mean = self.mean_head(x)
            x_disp = torch.exp(self.dispersion_head(x))
            return {"mean": x_mean, "disp": x_disp}
        

class TransformerModel(nn.Module):

    def __init__(self, token_dict: dict, label_dict: dict, model_config: ModelConfig, device: torch.device, seed: int = 0):
        """
        token_dim: the token dimensions, 5120 in UCE, following ESM2
        n_embed: the transformed token dimensions after the embedding, 1280 in UCE
        n_head: number of the attention head, 20 in UCE
        d_hid: the hidden layer dimensions, 5120 in UCE 
        n_layers: the number of layers, 4 layers in UCE github, in paper 33 layers
        output_dim: the output dimensions
        dropout: dropout rate
        """
        super().__init__()

        set_seed(seed)

        self.model_type = 'Transformer'
        self.model_config = model_config
        self.device = device

        # ------------------ Batch enc ------------------
        if self.model_config.batch_enc is not None:
            self.cond_embed = 64

            if self.model_config.batch_enc["name"] == "cat_pool":
                # categorical version
                self.batch_encoder = base_model.batch_encoder_catpool(n_cat_list = model_config.batch_enc["n_cat_list"], n_embed = self.model_config.d_embed)

            elif self.model_config.batch_enc["name"] == "cat_concat":
                # categorical version
                self.batch_encoder = base_model.batch_encoder_cat(n_cat_list = model_config.batch_enc["n_cat_list"], n_embed = self.model_config.d_embed, n_output = self.model_config.d_embed, p_drop = 0.0)

            self.batch_proj = nn.Sequential(nn.Linear(self.model_config.d_embed, self.cond_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.cond_embed))

        else:
            self.cond_embed = 0

        # ------------------ GenePool ------------------
        token_embed = token_dict["gene_embed"].to(self.model_config.precision)
        cls_embed = token_dict["cls_embed"].to(self.model_config.precision)
        self.token_embed = token_embed.to(self.device)
        self.cls_embed = cls_embed.to(self.device)

        self.n_meta = 256
        self.n_genes = token_embed.shape[0]

        self.gene_compression = graph_pool.GenePoolVanilla(A = token_dict["token_graph"].to(device), n_meta = self.n_meta, s_init = None, temp = token_dict["temp"]).to(device)
        self.gene_compression.load_state_dict(token_dict["gpoolvanilla"])
        for param in self.gene_compression.parameters():
            param.requires_grad = False

        if self.model_config.mlm_type == "meta":
            self.mask_embed = nn.Parameter(torch.randn((1, self.model_config.d_embed)))
            self.gene_decompression = nn.Identity()
        
        # ---------------- Transformer -------------------

        # gene name encoder, input: dimension of gene name embedding, token_dim
        # output: the n_embed, adopted from UCE
        self.gene_encoder = nn.Sequential(nn.Linear(token_embed.shape[1], self.model_config.d_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.model_config.d_embed))
                                                      
        self.expr_encoder = nn.Sequential(nn.Linear(32, self.model_config.d_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.model_config.d_embed))
        

        self.model_config.use_flashatten = use_flashatten

        if self.model_config.use_flashatten:
            # NOTE: need to use mixed precision
            self.transformer_encoder = flash_model.TransformerBlocks(d_model = self.model_config.d_embed,
                                                                    n_head = self.model_config.n_head,
                                                                    num_layers = self.model_config.n_layer,
                                                                    dim_feedforward = self.model_config.d_hidden,
                                                                    dropout = self.model_config.dropout,
                                                                    activation = "gelu")

        else:            
            self.transformer_encoder = base_model.TransformerBlocks(d_model = self.model_config.d_embed,
                                                                    n_head = self.model_config.n_head,
                                                                    num_layers = self.model_config.n_layer,
                                                                    dim_feedforward = self.model_config.d_hidden,
                                                                    dropout = self.model_config.dropout,
                                                                    activation = "gelu")

        # v3. even larger latent space
        self.decoder = nn.Sequential(base_model.full_block(self.model_config.d_embed, self.model_config.d_embed, self.model_config.dropout),
                                     nn.Linear(self.model_config.d_embed, self.model_config.d_output))


        if self.model_config.mlm_type == "meta":
            self.expr_predictor_meta = ConditionalPredictor(input_dim = self.model_config.d_output, condition_dim = self.cond_embed,
                                                    output_dim = self.n_meta, n_layers = 4, dropout = self.model_config.dropout, deep_inj = True)
                
        else:
            self.expr_predictor = ConditionalPredictor(input_dim = self.model_config.d_output, condition_dim = self.cond_embed,
                                                        output_dim = self.n_genes, n_layers = 6, dropout = self.model_config.dropout, deep_inj = True, nb_head = True)



        self.label_dict = label_dict
        self.label_dict["label_code"] = self.label_dict["label_bincode"].index.values
        self.label_unknown = np.where(self.label_dict["label_code"] == "unknown--unknown")[0][0]
        self.label_dict["label_bincode"] = torch.tensor(self.label_dict["label_bincode"].values).to(torch.float32).to(self.device)

        if model_config.use_contr:
            self.contrastive_label_mtx = self.calculate_contrastive_label()
        
        self.to(self.device)


    # NOTE: label_bincode has issue CL:0000192 (89) Tcell have the same bincode as CL:0000514 (618) Bcell
    def calculate_contrastive_label(self):
        print("calculate the positive, negative, and neutral label for each label...")
        label_bincode = self.label_dict["label_bincode"]
        assert torch.all(label_bincode[self.label_unknown] == 0)
        ancestor_matrix = contrastive.find_ancestor(label_bincode, label_bincode, chunk_size = 16)
        # ancestor_matrix = torch.hstack([label_bincode, torch.ones(label_bincode.shape[0], 1).to(label_bincode.device)])
        descendent_matrix = contrastive.find_descend(label_bincode, label_bincode)
        # descendent_matrix = ancestor_matrix.T.clone()
        # remove same label, keep only strict ancestor
        # equal_matrix = contrastive.exact_equal(label_bincode, label_bincode, chunk_size = 16)
        equal_matrix = torch.eye(descendent_matrix.shape[0]).bool().to(label_bincode.device)
        neutral_matrix = (ancestor_matrix & ~equal_matrix)

        # NOTE: for the unknown cell, no descendent/positive, no negative, all neutral (no loss applied)
        descendent_matrix[self.label_unknown, :] = False
        neutral_matrix[self.label_unknown, :] = True

        contr_label_matrix = -1 * torch.ones_like(descendent_matrix).to(label_bincode.device)
        # descendent are positive samples
        contr_label_matrix[descendent_matrix] = 1
        contr_label_matrix[neutral_matrix] = 0
        print("Done.")

        return contr_label_matrix


    def freeze_fm_gradient(self, freeze_trans: bool, freeze_batchenc: bool, freeze_compression: bool):
        self.compression_grad = not freeze_compression
        self.batchenc_grad = not freeze_batchenc
        self.trans_grad = not freeze_trans

        if self.model_config.batch_enc is not None:
            for param in self.batch_encoder.parameters():
                param.requires_grad = self.batchenc_grad
            for param in self.batch_proj.parameters():
                param.requires_grad = self.batchenc_grad

        for param in self.gene_compression.parameters():
            param.requires_grad = self.compression_grad
        # the predictor is freeze and unfreeze following compression model
        if self.model_config.mlm_type != "meta":
            for param in self.expr_predictor.parameters():
                param.requires_grad = self.compression_grad

        for param in self.transformer_encoder.parameters():
            param.requires_grad = self.trans_grad
        for param in self.decoder.parameters():
            param.requires_grad = self.trans_grad
        for param in self.expr_encoder.parameters():
            param.requires_grad = self.trans_grad
        for param in self.gene_encoder.parameters():
            param.requires_grad = self.trans_grad
        if self.model_config.mlm_type == "meta":
            self.mask_embed.requires_grad = self.trans_grad
                

    def forward(self, counts_norm: torch.Tensor):
        """
        Parameters:
        --------------
            expr_sent: the gene expression sentence, of the shape (n_batchs, n_tokens)
        """
        
        if self.model_config.mlm_type != "meta":
            # mask added on gene
            mask_prob = torch.full(counts_norm.shape, self.model_config.mask_prob).to(self.device)
            mask_gene = torch.bernoulli(mask_prob).bool()             
            counts_norm = counts_norm.masked_fill(mask_gene, 0)

        # reconstruct meta-gene expr, mask is added on meta-gene
        S, token_embed_meta, counts_norm_meta = self.gene_compression(gene_embed = self.token_embed, expr = counts_norm, log_norm = True)
        gene_embed = torch.vstack([self.cls_embed, token_embed_meta])
        gene_embed = self.gene_encoder(gene_embed).unsqueeze(0).repeat(counts_norm.shape[0], 1, 1)     

        # construct the gene expression sentence (match the word embedding)
        expr_sent = torch.hstack([torch.zeros((counts_norm_meta.shape[0], 1)).to(counts_norm_meta.device), counts_norm_meta])
        expr_embed = base_model.fourier_positional_encoding(expr_sent.unsqueeze(2), embedding_dim = 32)
        expr_embed = self.expr_encoder(expr_embed)

        # insert condition, make (258) tokens, sentence composition [cls, expr1, expr2, ..., exprm, cond]
        gene_sent_embed = gene_embed * math.sqrt(self.model_config.d_embed)
        expr_sent_embed = expr_embed * math.sqrt(self.model_config.d_embed)
        
        # updated mask
        if self.model_config.mlm_type == "meta":
            # add mask to the input data
            mask_prob = torch.full(gene_sent_embed.shape[:2], self.model_config.mask_prob).to(self.device)
            # do not add mask on cls and cond positions
            mask_prob[:, 0] = 0
            if self.model_config.insert_transformer:
                # masking the last token (batch token) if insert_transformer is True (batch token not exist)
                mask_prob[:, -1] = 0

            # sample mask using bernoulli
            mask = torch.bernoulli(mask_prob).bool()
            mask_embed = self.mask_embed.view(1, 1, -1)
            expr_sent_embed = torch.where(mask.unsqueeze(-1), mask_embed, expr_sent_embed)

            if self.model_config.insert_transformer:
                mask_gene = mask[:, 1:-1]
            else:
                mask_gene = mask[:, 1:]

        embed = self.transformer_encoder((gene_sent_embed + expr_sent_embed).permute(1, 0, 2), src_key_padding_mask = None)
        embed = self.decoder(embed)
        cell_embed = embed[0, :, :]
        cell_embed = nn.functional.normalize(cell_embed, dim=1) # Normalize.
        return embed, cell_embed, mask_gene


    def predict_expr(self, cell_embed: torch.Tensor, batch_factors: torch.Tensor):

        # incorporate the batch condition
        if self.model_config.batch_enc is not None:
            batch_embed, _ = self.batch_encoder(batch_factors = batch_factors)
            batch_embed = self.batch_proj(batch_embed)
        
        else:
            batch_embed = None
        
        if self.model_config.mlm_type == "meta":
            expr_pred_meta = self.expr_predictor_meta(x = cell_embed, cond = batch_embed)
            expr_pred = None
        else:
            expr_pred_meta = None
            expr_pred = self.expr_predictor(x = cell_embed, cond = batch_embed)
        return expr_pred, expr_pred_meta



