import torch 
from torch import nn
import torch.nn.functional as F
from .models import Norm


# class PermuteIn(nn.Module):
#     def __init__(self, 
#                  dim,
#                  heads,
#                  mode="window", 
#                  roll=0.4,
#                  chunks=4, 
#                  ):
#         super().__init__()
#         full_dim = dim * heads
#         self.mode = mode
#         self.permute = None
      
#         self.sliding_windows = nn.ParameterList([
#             nn.Parameter(torch.randn(512))
#             for _ in range(5*5)
#         ])
#         self.merge_windows = nn.ModuleList([
#             nn.Linear(512*5, 512)
#             for _ in range(5)
#         ])
#         self.norm = nn.LayerNorm(512)
        
#     def forward(self, x):            
#         x = x.reshape(x.shape[0], -1)
#         B, L = x.shape
#         if L % (512) != 0:
#             pad = 512 - L % 512
#             x = F.pad(x, (0, pad))
#         for hirerachy in range(5):
#             interimx = []
#             for i in range(5):
#                 _x = x.view(x.shape[0], -1, 512)
#                 _x = _x * self.sliding_windows[hirerachy*5 + i]
#                 _x = self.norm(_x)
#                 _x = F.silu(_x)
#                 interimx.append(_x)
#             x = torch.stack(interimx, dim=2)

#             x = x.reshape (x.shape[0], -1, 512*5)
#             x = self.merge_windows[hirerachy](x)
#             x = self.norm(x)
#             x = x.reshape(x.shape[0], -1)
            
#         # unpad
#         if L % (512) != 0:
#             x = x[:, :-pad]
        
#         return x
    


# class PermuteIn(nn.Module):
#     def __init__(self, 
#                  dim,
#                  heads,
#                  mode="window", 
#                  roll=0.4,
#                  chunks=4, 
#                  ):
#         super().__init__()
#         full_dim = dim * heads
    
      
#         self.sliding_windows = nn.ParameterList([
#             nn.Parameter(torch.randn(512))
#             for _ in range(5*4)
#         ])
#         self.merge_windows = nn.ModuleList([
#             nn.Linear(512*5, 512)
#             for _ in range(5)
#         ])

#         self.norm = nn.LayerNorm(512)
        
#     def forward(self, x):
#         x = x.reshape(x.shape[0], -1)
#         B, L = x.shape
        
#         if L % (512) != 0:
#             pad = 512 - L % 512
#             x = F.pad(x, (0, pad))
#         for hirerachy in range(4):
#             interimx = []
#             for i in range(5):
#                 _x = x.view(x.shape[0], -1, 512)
#                 _x = F.relu(_x) * self.sliding_windows[hirerachy*5 + i]
#                 _x = self.norm(_x)
#                 # _x = F.relu(_x)
#                 interimx.append(_x)
#             _x = torch.stack(interimx, dim=2)
#             _x = _x.reshape (x.shape[0], -1, 512*5)
#             x = self.merge_windows[hirerachy](_x) + x.view(x.shape[0], -1, 512)
#             x = self.norm(x)
#             # _x = F.relu(_x)
#             x = x.reshape(x.shape[0], -1)
            
#         # unpad
#         if L % (512) != 0:
#             x = x[:, :-pad]

#         return x
    

class Unpermute(nn.Module):

    def __init__(self, indices):
        super().__init__()
        perm_matrix = F.one_hot(indices, num_classes=indices.shape[0]).float()
        unperm_matrix = perm_matrix.inverse()
        unperm = unperm_matrix.argmax(dim=-1).long()
        self.register_buffer("unperm", unperm)

    def forward(self, x):
        return x[:, self.unperm]

class BiasAdd(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias

class SparseLinear(nn.Module):
    
    def __init__(self, in_dim=64, out_dim=64, heads=8, bias=True):
        super(SparseLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h = heads
        self.weight = nn.Parameter(torch.randn(heads, in_dim, out_dim))
        self.bias_add = BiasAdd(out_dim) if bias else nn.Identity()

    def forward(self, x):
        # b, h * in_dim
        b, h, in_dim = x.shape[0], self.h, self.in_dim
        x = x.reshape(b, h, in_dim)#.reshape(b, h, -1, in_dim)
        x = torch.einsum('bhd,hdl->bhl', x, self.weight)#torch.vmap(lambda x: torch.bmm(x, self.weight))(x)
        x = x.squeeze(-2).reshape(b, h , self.out_dim)
        x = self.bias_add(x)
        x = x.reshape(b, h * self.out_dim)
        return x


class SparseMLP(nn.Module):

    def __init__(self, dim=64, 
                        heads=8, 
                        act=nn.GELU, 
                        mlp_dim=256, 
                        unperm=False, 
                        residual=False, 
                        dropout=0., 
                        permute_mode="chunk_random", # ["random", "roll", "chunk_random", "linear"]
                        ):
        super(SparseMLP, self).__init__()
        self.d = dim
        self.h = heads
        self.residual = lambda x, y: x + y if residual else x

        self.up = nn.Parameter(torch.randn(heads, dim, mlp_dim))
        self.down = nn.Parameter(torch.randn(heads, mlp_dim, dim))
        self.act = act()

        self.unperm = nn.Identity()
        if permute_mode != "linear":
            self.perm = PermuteIn(dim, heads, mode=permute_mode)
            if unperm:
                self.unperm = Unpermute(self.perm.permute)
        else:
            self.perm = nn.Linear(dim * heads, dim * heads)
            if unperm:
                self.unperm = nn.Linear(dim * heads, dim * heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        b, h, d = x.shape[0], self.h, self.d

        x = self.perm(x) # reorder features to have different interactions
        x = x.reshape(b, h, d).reshape(b, h, -1, d) # b*h, 1, i bmm is annoying this way
        # x = torch.bmm(x, self.up.repeat(b, 1, 1))
        x = torch.vmap(lambda x: torch.bmm(x, self.up))(x)
        x = self.act(x)
        x = self.dropout(x)
        # x = torch.bmm(x, self.down.repeat(b, 1, 1)) # b*h, 1, d
        x = torch.vmap(lambda x: torch.bmm(x, self.down))(x)
        x = self.dropout(x)
        x = x.squeeze(-2).reshape(b, h * d)
        x = self.unperm(x)

        x = self.residual(x, residual)

        return x




class SillyNormAct(torch.nn.Module):

    def __init__(self, dim, norm_in_type="layer", norm_out_type="none", act=torch.nn.GELU):
        super().__init__()
        self.norm_in = Norm(dim, norm_in_type) if norm_in_type != "none" else nn.Identity()
        self.act = act()
        self.norm_out = Norm(dim, norm_out_type) if norm_out_type != "none" else nn.Identity()

    def forward(self, x):
        return self.norm_out(self.act(self.norm_in(x)))





