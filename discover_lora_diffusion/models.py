import torch
from torch import nn
import math
import torch
import torch.nn as nn
import numpy as np
import math
import sys
import torch.nn.functional as F


sys.path.append('..')
# from common.sparse import SparseLinear, PermuteIn
from common.models import TimestepEmbedding, AdaNorm, ChunkFanOut, Attention, FeedForward, DiTBlock, Resnet


class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        ada_dim: int = 512,
        cond_dim: int = 512,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        self.norm1 = AdaNorm(in_dim, ada_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.norm2 = AdaNorm(mid_dim, ada_dim)
        self.cond_in = AdaIN(mid_dim, cond_dim)
        self.linear_fm = nn.Linear(mid_dim, mid_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.act = act()


    def forward(
        self,
        hidden_states,
        ada_emb=None,
        face_embed=None,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.linear1(self.act(self.norm1(hidden_states, ada_emb)))
        if face_embed is not None:
            hidden_states = self.linear_fm(self.act(self.cond_in(hidden_states, face_embed))) # this will corrupt the data so much, because it's not trained
        hidden_states = self.linear2(self.dropout(self.act(self.norm2(hidden_states, ada_emb))))
      

        return hidden_states + resid


class ResnetBlock(nn.Module):

    def __init__(self, num_layers=3, in_dim=256, mid_dim=256, ada_dim=512, cond_dim=512):
        super().__init__()
        self.layers = nn.ModuleList([Resnet(in_dim, mid_dim, ada_dim=ada_dim, cond_dim=cond_dim) for _ in range(num_layers)])

    def forward(self, x, ada_emb,  face_embed=None):
        for layer in self.layers:
            x = layer(x, ada_emb,  face_embed)
        return x


class AdaIN(nn.Module):
    def __init__(self, mid_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(num_features, mid_dim * 2)
        self.combine = nn.Sequential(
            # nn.SiLU(),
            nn.Linear(mid_dim, mid_dim),
        )

    def forward(self, _x, ada_emb):
        # Normalize input
        x = self.norm(_x)
        # Get scaling and bias parameters from adaptive embedding
        ada_params = self.fc(ada_emb)
        gamma, beta = ada_params.chunk(2, dim=-1)
        # Reshape for broadcasting
        # Apply AdaIN
        return ((1.0 + gamma) * x + beta) #+ _x


# class CrossAttention(nn.Module):
#     def __init__(self, dim):
#         super(CrossAttention, self).__init__()
#         self.query_proj = nn.Linear(dim, dim)
#         self.key_proj = nn.Linear(dim, dim)
#         self.value_proj = nn.Linear(dim, dim)

#     def forward(self, emb1, emb2):
#         # emb1 and emb2 are [batch_size, num_features, dim]
#         query = self.query_proj(emb1)
#         key = self.key_proj(emb2)
#         value = self.value_proj(emb2)

#         # Compute attention scores
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
#         attention = F.softmax(scores, dim=-1)

#         # Apply attention to values
#         attended = torch.matmul(attention, value)

#         # Combine attended features back with the original embedding
#         combined = attended + emb1
#         return combined


class LoraDiffusion(torch.nn.Module):

    def __init__(self, data_dim=1_365_504, 
                    model_dim=256, 
                    ff_mult=3, 
                    chunks=1, 
                    act=torch.nn.SiLU, 
                    num_blocks=4, 
                    layers_per_block=3,
                      cond_dropout_prob=0.3 ,
                      cond_dim=512
                    ):
        super().__init__()
        self.cond_dropout_prob = cond_dropout_prob
        self.time_embed = TimestepEmbedding(model_dim//2, model_dim//2)
        self.cond_dim = cond_dim

        self.in_norm = nn.LayerNorm(data_dim)
        self.in_proj = nn.Linear(data_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, data_dim)
        self.out_norm = nn.LayerNorm(model_dim)

        # self.out_proj.weight.data = self.in_proj.weight.data.T


        self.cond_learn = nn.Linear(cond_dim , cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.cond_out_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, cond_dim),
        )

        self.conditioning = None

        self.downs = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim, in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])
        self.ups = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim,  in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])

    def get_conditioning(self, face_embeddings):
        if face_embeddings is not None:
            face_emb = face_embeddings.detach()
            # _, face_emb = face_emb.chunk(2, dim=-1)
            # print("face_emb", face_emb.shape)
            # full_emb = F.normalize(full_emb, p=2, dim=-1)
            # face_emb = F.normalize(face_emb, p=2, dim=-1)

            # face_emb = torch.cat([full_emb, face_emb], dim=-1)
            # face_emb = self.cond_learn(face_emb)
            # face_emb = self.cond_norm(face_emb)
            self.conditioning = face_emb
        else:
            self.conditioning = None


    def reconstruct_conditioning(self, x):
        # pred_cond = self.cond_out(x)
        embedding = self.cond_out_proj(x)

        return embedding



    def forward(self, x, t, face_embeddings=None):
        ada_emb = self.time_embed(t)
        
        self.get_conditioning(face_embeddings)
        # lora_dim = 10_000
        x = self.in_norm(x)
        x = self.in_proj(x)
        # x_skip = x
        skips = []
        for down in (self.downs):
      
            x = down(x, ada_emb, self.conditioning)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips)):
      
            x = up(x, ada_emb,  self.conditioning) + skip


        if face_embeddings is not None:
            pred_cond = self.reconstruct_conditioning(x)
        else:
            pred_cond = None

        x = self.out_norm(x) #+ x_skip
        x = self.out_proj(x)
        
        return x, pred_cond


class DiT(nn.Module):
    def __init__(
        self,
        total_dim=1_365_504,
        dim = 1536,
        num_tokens=889,
        time_dim = 384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens

        self.proj_in = torch.nn.Linear(dim, dim)
        self.time_embed =  TimestepEmbedding(time_dim, time_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([DiTBlock(dim, time_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True)
        self.proj_out = nn.Linear(dim, dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        x = x.reshape(x.size(0), -1, self.dim)
        x = self.proj_in(x) + self.pos_embed.expand(x.size(0), -1, -1)
        t = self.time_embed(t)
        for block in self.blocks:
            x = block(x, t)
        x = self.norm_out(x)
        x = self.proj_out(x)
        x = x.reshape(x.size(0), -1)
        return x






#