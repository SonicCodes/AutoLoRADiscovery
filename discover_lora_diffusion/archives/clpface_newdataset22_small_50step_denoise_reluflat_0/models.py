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
        self.cond_in = HyperAda(mid_dim, cond_dim)
        self.linear_fm = nn.Linear(mid_dim, mid_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.linear_mlp = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.act = act()


    def forward(
        self,
        hidden_states,
        ada_emb=None,
        face_embed=None,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.linear1(self.act(self.norm1(hidden_states, ada_emb)))
        hidden_states = self.linear_mlp(hidden_states)
        if face_embed is not None:
            hidden_states = self.linear_fm(self.act(self.cond_in(hidden_states, face_embed))) # this will corrupt the data so much, because it's not trained
        hidden_states = self.linear2(hidden_states)
      

        return hidden_states + resid


class ResnetBlock(nn.Module):

    def __init__(self, num_layers=3, in_dim=256, mid_dim=256, ada_dim=512, cond_dim=512, out_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([Resnet(in_dim, mid_dim, ada_dim=ada_dim, cond_dim=cond_dim, out_dim=out_dim) for _ in range(num_layers)])
        # self.out_project = nn.Linear(in_dim, out_dim if out_dim is not None else in_dim)
    def forward(self, x, ada_emb,  face_embed=None):
        for layer in self.layers:
            x = layer(x, ada_emb,  face_embed)
        return x
        # return x


class AdaIN(nn.Module):
    def __init__(self, mid_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(num_features, mid_dim * 2)

    def forward(self, _x, ada_emb):
        x = self.norm(_x)
        ada_params = self.fc(ada_emb)
        gamma, beta = ada_params.chunk(2, dim=-1)
        return ((1.0 + gamma) * x + beta)

class HyperAda(nn.Module):
    def __init__(self, mid_dim, num_features, rank=16):
        super().__init__()
        # self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.rank = rank
        self.generate_weights = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.SiLU(),
            nn.Linear(64, mid_dim * rank * 2)
        )
        # self.out_norm = nn.LayerNorm(mid_dim)

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape
        ada_emb = ada_emb.reshape(B, -1)
        x_weights = self.generate_weights(ada_emb)
        x_down, x_up = x_weights.view(-1, D, self.rank, 2).chunk(2, dim=-1)
        x_down, x_up = x_down.squeeze(-1), x_up.squeeze(-1)
        # print(x_down.shape, x.shape)
        x = torch.einsum('bc,bco->bo', x, x_down)
        x = torch.einsum('bc,bco->bo', x, x_up.permute(0, 2, 1))
        
        x = x.reshape(B, D)

        # x = self.out_norm(x)

        return x + x_in

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


config = [('down_blocks.0/attn1/to_q.lora_A', 0, 320, 320, 0.03346851095557213),
 ('down_blocks.0/attn1/to_q.lora_B', 320, 640, 320, 0.004553175065666437),
 ('down_blocks.0/attn1/to_v.lora_A', 640, 960, 320, 0.03245671093463898),
 ('down_blocks.0/attn1/to_v.lora_B', 960, 1280, 320, 0.0035195741802453995),
 ('down_blocks.0/attn2/to_q.lora_A', 1280, 1600, 320, 0.0314430370926857),
 ('down_blocks.0/attn2/to_q.lora_B', 1600, 1920, 320, 0.004563907161355019),
 ('down_blocks.0/attn2/to_v.lora_A', 1920, 2688, 768, 0.021910294890403748),
 ('down_blocks.0/attn2/to_v.lora_B', 2688, 3008, 320, 0.003337073838338256),
 ('down_blocks.0/attn1/to_q.lora_A', 3008, 3328, 320, 0.03344935178756714),
 ('down_blocks.0/attn1/to_q.lora_B', 3328, 3648, 320, 0.0036399837117642164),
 ('down_blocks.0/attn1/to_v.lora_A', 3648, 3968, 320, 0.03341167792677879),
 ('down_blocks.0/attn1/to_v.lora_B', 3968, 4288, 320, 0.002928160596638918),
 ('down_blocks.0/attn2/to_q.lora_A', 4288, 4608, 320, 0.032661180943250656),
 ('down_blocks.0/attn2/to_q.lora_B', 4608, 4928, 320, 0.0031401854939758778),
 ('down_blocks.0/attn2/to_v.lora_A', 4928, 5696, 768, 0.0247409138828516),
 ('down_blocks.0/attn2/to_v.lora_B', 5696, 6016, 320, 0.005309948232024908),
 ('down_blocks.1/attn1/to_q.lora_A', 6016, 6656, 640, 0.023771541193127632),
 ('down_blocks.1/attn1/to_q.lora_B', 6656, 7296, 640, 0.0043926299549639225),
 ('down_blocks.1/attn1/to_v.lora_A', 7296, 7936, 640, 0.02421427331864834),
 ('down_blocks.1/attn1/to_v.lora_B', 7936, 8576, 640, 0.003957018721848726),
 ('down_blocks.1/attn2/to_q.lora_A', 8576, 9216, 640, 0.023453030735254288),
 ('down_blocks.1/attn2/to_q.lora_B', 9216, 9856, 640, 0.004457980860024691),
 ('down_blocks.1/attn2/to_v.lora_A', 9856, 10624, 768, 0.02108289673924446),
 ('down_blocks.1/attn2/to_v.lora_B', 10624, 11264, 640, 0.0044920737855136395),
 ('down_blocks.1/attn1/to_q.lora_A', 11264, 11904, 640, 0.023866992443799973),
 ('down_blocks.1/attn1/to_q.lora_B', 11904, 12544, 640, 0.004428502172231674),
 ('down_blocks.1/attn1/to_v.lora_A', 12544, 13184, 640, 0.02398420311510563),
 ('down_blocks.1/attn1/to_v.lora_B', 13184, 13824, 640, 0.0035206947941333055),
 ('down_blocks.1/attn2/to_q.lora_A', 13824, 14464, 640, 0.024173414334654808),
 ('down_blocks.1/attn2/to_q.lora_B', 14464, 15104, 640, 0.006062699016183615),
 ('down_blocks.1/attn2/to_v.lora_A', 15104, 15872, 768, 0.02167080156505108),
 ('down_blocks.1/attn2/to_v.lora_B', 15872, 16512, 640, 0.004011091310530901),
 ('down_blocks.2/attn1/to_q.lora_A', 16512, 17792, 1280, 0.017633598297834396),
 ('down_blocks.2/attn1/to_q.lora_B', 17792, 19072, 1280, 0.005403602495789528),
 ('down_blocks.2/attn1/to_v.lora_A', 19072, 20352, 1280, 0.0196865051984787),
 ('down_blocks.2/attn1/to_v.lora_B', 20352, 21632, 1280, 0.004640033468604088),
 ('down_blocks.2/attn2/to_q.lora_A', 21632, 22912, 1280, 0.017589913681149483),
 ('down_blocks.2/attn2/to_q.lora_B', 22912, 24192, 1280, 0.005873133894056082),
 ('down_blocks.2/attn2/to_v.lora_A', 24192, 24960, 768, 0.02330712042748928),
 ('down_blocks.2/attn2/to_v.lora_B', 24960, 26240, 1280, 0.003336850320920348),
 ('down_blocks.2/attn1/to_q.lora_A', 26240, 27520, 1280, 0.018302295356988907),
 ('down_blocks.2/attn1/to_q.lora_B', 27520, 28800, 1280, 0.005325611215084791),
 ('down_blocks.2/attn1/to_v.lora_A', 28800, 30080, 1280, 0.0191669799387455),
 ('down_blocks.2/attn1/to_v.lora_B', 30080, 31360, 1280, 0.00459354929625988),
 ('down_blocks.2/attn2/to_q.lora_A', 31360, 32640, 1280, 0.01752602681517601),
 ('down_blocks.2/attn2/to_q.lora_B',
  32640,
  33920,
  1280,
  0.0060758027248084545),
 ('down_blocks.2/attn2/to_v.lora_A', 33920, 34688, 768, 0.02241193689405918),
 ('down_blocks.2/attn2/to_v.lora_B',
  34688,
  35968,
  1280,
  0.0030942053999751806),
 ('mid_block.attentions/attn1/to_q.lora_A',
  35968,
  37248,
  1280,
  0.01754344068467617),
 ('mid_block.attentions/attn1/to_q.lora_B',
  37248,
  38528,
  1280,
  0.005082845222204924),
 ('mid_block.attentions/attn1/to_v.lora_A',
  38528,
  39808,
  1280,
  0.01952926628291607),
 ('mid_block.attentions/attn1/to_v.lora_B',
  39808,
  41088,
  1280,
  0.004893908277153969),
 ('mid_block.attentions/attn2/to_q.lora_A',
  41088,
  42368,
  1280,
  0.017860369756817818),
 ('mid_block.attentions/attn2/to_q.lora_B',
  42368,
  43648,
  1280,
  0.006144731305539608),
 ('mid_block.attentions/attn2/to_v.lora_A',
  43648,
  44416,
  768,
  0.024062925949692726),
 ('mid_block.attentions/attn2/to_v.lora_B',
  44416,
  45696,
  1280,
  0.0037334468215703964),
 ('up_blocks.1/attn1/to_q.lora_A', 45696, 46976, 1280, 0.018411893397569656),
 ('up_blocks.1/attn1/to_q.lora_B', 46976, 48256, 1280, 0.005813125055283308),
 ('up_blocks.1/attn1/to_v.lora_A', 48256, 49536, 1280, 0.020095257088541985),
 ('up_blocks.1/attn1/to_v.lora_B', 49536, 50816, 1280, 0.0048658824525773525),
 ('up_blocks.1/attn2/to_q.lora_A', 50816, 52096, 1280, 0.017398452386260033),
 ('up_blocks.1/attn2/to_q.lora_B', 52096, 53376, 1280, 0.005977979861199856),
 ('up_blocks.1/attn2/to_v.lora_A', 53376, 54144, 768, 0.023836299777030945),
 ('up_blocks.1/attn2/to_v.lora_B', 54144, 55424, 1280, 0.0031119389459490776),
 ('up_blocks.1/attn1/to_q.lora_A', 55424, 56704, 1280, 0.019207481294870377),
 ('up_blocks.1/attn1/to_q.lora_B', 56704, 57984, 1280, 0.005992804653942585),
 ('up_blocks.1/attn1/to_v.lora_A', 57984, 59264, 1280, 0.021595565602183342),
 ('up_blocks.1/attn1/to_v.lora_B', 59264, 60544, 1280, 0.005385051015764475),
 ('up_blocks.1/attn2/to_q.lora_A', 60544, 61824, 1280, 0.017851797863841057),
 ('up_blocks.1/attn2/to_q.lora_B', 61824, 63104, 1280, 0.005849853157997131),
 ('up_blocks.1/attn2/to_v.lora_A', 63104, 63872, 768, 0.023986639454960823),
 ('up_blocks.1/attn2/to_v.lora_B', 63872, 65152, 1280, 0.003360884264111519),
 ('up_blocks.1/attn1/to_q.lora_A', 65152, 66432, 1280, 0.01880822144448757),
 ('up_blocks.1/attn1/to_q.lora_B', 66432, 67712, 1280, 0.005930751096457243),
 ('up_blocks.1/attn1/to_v.lora_A', 67712, 68992, 1280, 0.019831882789731026),
 ('up_blocks.1/attn1/to_v.lora_B', 68992, 70272, 1280, 0.0051866332069039345),
 ('up_blocks.1/attn2/to_q.lora_A', 70272, 71552, 1280, 0.01796850375831127),
 ('up_blocks.1/attn2/to_q.lora_B', 71552, 72832, 1280, 0.005855177529156208),
 ('up_blocks.1/attn2/to_v.lora_A', 72832, 73600, 768, 0.02155807986855507),
 ('up_blocks.1/attn2/to_v.lora_B', 73600, 74880, 1280, 0.002725658705458045),
 ('up_blocks.2/attn1/to_q.lora_A', 74880, 75520, 640, 0.02507838048040867),
 ('up_blocks.2/attn1/to_q.lora_B', 75520, 76160, 640, 0.005619049537926912),
 ('up_blocks.2/attn1/to_v.lora_A', 76160, 76800, 640, 0.024947481229901314),
 ('up_blocks.2/attn1/to_v.lora_B', 76800, 77440, 640, 0.004646731074899435),
 ('up_blocks.2/attn2/to_q.lora_A', 77440, 78080, 640, 0.024707071483135223),
 ('up_blocks.2/attn2/to_q.lora_B', 78080, 78720, 640, 0.007063147611916065),
 ('up_blocks.2/attn2/to_v.lora_A', 78720, 79488, 768, 0.022415513172745705),
 ('up_blocks.2/attn2/to_v.lora_B', 79488, 80128, 640, 0.003970491699874401),
 ('up_blocks.2/attn1/to_q.lora_A', 80128, 80768, 640, 0.0273636095225811),
 ('up_blocks.2/attn1/to_q.lora_B', 80768, 81408, 640, 0.006009900942444801),
 ('up_blocks.2/attn1/to_v.lora_A', 81408, 82048, 640, 0.025179849937558174),
 ('up_blocks.2/attn1/to_v.lora_B', 82048, 82688, 640, 0.004426096100360155),
 ('up_blocks.2/attn2/to_q.lora_A', 82688, 83328, 640, 0.025323769077658653),
 ('up_blocks.2/attn2/to_q.lora_B', 83328, 83968, 640, 0.007592460606247187),
 ('up_blocks.2/attn2/to_v.lora_A', 83968, 84736, 768, 0.029621664434671402),
 ('up_blocks.2/attn2/to_v.lora_B', 84736, 85376, 640, 0.005635158158838749),
 ('up_blocks.2/attn1/to_q.lora_A', 85376, 86016, 640, 0.025046488270163536),
 ('up_blocks.2/attn1/to_q.lora_B', 86016, 86656, 640, 0.005727486219257116),
 ('up_blocks.2/attn1/to_v.lora_A', 86656, 87296, 640, 0.024599524214863777),
 ('up_blocks.2/attn1/to_v.lora_B', 87296, 87936, 640, 0.004428410902619362),
 ('up_blocks.2/attn2/to_q.lora_A', 87936, 88576, 640, 0.024861536920070648),
 ('up_blocks.2/attn2/to_q.lora_B', 88576, 89216, 640, 0.00872720591723919),
 ('up_blocks.2/attn2/to_v.lora_A', 89216, 89984, 768, 0.02369890734553337),
 ('up_blocks.2/attn2/to_v.lora_B', 89984, 90624, 640, 0.004600498825311661),
 ('up_blocks.3/attn1/to_q.lora_A', 90624, 90944, 320, 0.03510580211877823),
 ('up_blocks.3/attn1/to_q.lora_B', 90944, 91264, 320, 0.005146631971001625),
 ('up_blocks.3/attn1/to_v.lora_A', 91264, 91584, 320, 0.03169950842857361),
 ('up_blocks.3/attn1/to_v.lora_B', 91584, 91904, 320, 0.0034357912372797728),
 ('up_blocks.3/attn2/to_q.lora_A', 91904, 92224, 320, 0.03277582302689552),
 ('up_blocks.3/attn2/to_q.lora_B', 92224, 92544, 320, 0.005698359105736017),
 ('up_blocks.3/attn2/to_v.lora_A', 92544, 93312, 768, 0.023317117244005203),
 ('up_blocks.3/attn2/to_v.lora_B', 93312, 93632, 320, 0.005420367699116468),
 ('up_blocks.3/attn1/to_q.lora_A', 93632, 93952, 320, 0.033382028341293335),
 ('up_blocks.3/attn1/to_q.lora_B', 93952, 94272, 320, 0.004492181818932295),
 ('up_blocks.3/attn1/to_v.lora_A', 94272, 94592, 320, 0.03257472813129425),
 ('up_blocks.3/attn1/to_v.lora_B', 94592, 94912, 320, 0.003115034895017743),
 ('up_blocks.3/attn2/to_q.lora_A', 94912, 95232, 320, 0.03203093260526657),
 ('up_blocks.3/attn2/to_q.lora_B', 95232, 95552, 320, 0.005714129656553268),
 ('up_blocks.3/attn2/to_v.lora_A', 95552, 96320, 768, 0.0230315662920475),
 ('up_blocks.3/attn2/to_v.lora_B', 96320, 96640, 320, 0.005082032177597284),
 ('up_blocks.3/attn1/to_q.lora_A', 96640, 96960, 320, 0.03439769521355629),
 ('up_blocks.3/attn1/to_q.lora_B', 96960, 97280, 320, 0.0046622673980891705),
 ('up_blocks.3/attn1/to_v.lora_A', 97280, 97600, 320, 0.0330582894384861),
 ('up_blocks.3/attn1/to_v.lora_B', 97600, 97920, 320, 0.0037263997364789248),
 ('up_blocks.3/attn2/to_q.lora_A', 97920, 98240, 320, 0.03359720855951309),
 ('up_blocks.3/attn2/to_q.lora_B', 98240, 98560, 320, 0.005731364246457815),
 ('up_blocks.3/attn2/to_v.lora_A', 98560, 99328, 768, 0.02355472557246685),
 ('up_blocks.3/attn2/to_v.lora_B', 99328, 99648, 320, 0.0052943299524486065)]

latent_std, latent_mean = torch.load("/home/ubuntu/AutoLoRADiscovery/latent_properties.pt")

class LoraDiffusion(torch.nn.Module):

    def __init__(self, data_dim=1_365_504, 
                    model_dim=256, 
                    ff_mult=3, 
                    chunks=1, 
                    act=torch.nn.SiLU, 
                    num_blocks=4, 
                    layers_per_block=3,
                      cond_dropout_prob=0.3 ,
                      cond_dim=40
                    ):
        super().__init__()
        # cond_dim = 128
        self.cond_dropout_prob = cond_dropout_prob
        self.time_embed = TimestepEmbedding(model_dim//2, model_dim//2)
        self.cond_dim = cond_dim
        latent_dim = model_dim*4
        self.input_dim = data_dim
        self.latent_dim = model_dim*4

        tiny_encoders = []
        tiny_decoders = [] # transpose of encoder
        for i, (_, _, _, dim, _) in enumerate(config):
            tiny_encoder = nn.Linear(dim, 320)
            tiny_decoder = nn.Linear(320, dim)
            # tiny_decoder.weight.data = tiny_encoder.weight.data.T
            tiny_encoders.append(tiny_encoder)
            tiny_decoders.append(tiny_decoder)

        self.tiny_encoders = nn.ModuleList(tiny_encoders)
        self.tiny_decoders = nn.ModuleList(tiny_decoders)

        # self.encoder_self_attention = nn.MultiheadAttention(320, 4, batch_first=True)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(320*len(tiny_encoders), model_dim),
        )

        # Decoder       
        self.decoder = nn.Sequential(
            nn.Linear(model_dim*2, 320*len(tiny_encoders)),
        )

        self.in_norm = nn.LayerNorm(data_dim)
        # self.in_proj = nn.Linear(data_dim, model_dim)
        # self.out_proj = nn.Linear(model_dim, data_dim)
        self.out_norm = nn.LayerNorm(model_dim*2)

        # self.out_proj.weight.data = self.in_proj.weight.data.T


        # self.cond_learn = nn.Sequential(
        #     nn.Linear(40, cond_dim),
        #     nn.SiLU(),
        #     # nn.Tanh()
        # )
        # self.cond_norm = nn.LayerNorm(cond_dim)

        self.cond_out_proj = nn.Sequential(
            nn.LayerNorm(model_dim*2),
            nn.Linear(model_dim*2, 40),
            nn.Tanh()
        )

        self.conditioning = None

        self.downs = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim, in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])
        self.ups = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim,  in_dim=model_dim, out_dim=model_dim, mid_dim=int((model_dim) * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])

        # self.hyper_net_gen = nn.Linear(cond_dim, 2048*32*2)


    def encode(self, x, kld=False):
        parts = []
        for (_, start, end, _, _), tiny_encoder in zip(config, self.tiny_encoders):
            part = tiny_encoder(x[:, start:end])
            parts.append(part)
        x = torch.stack(parts, dim=1)
        # x_atten, x_atten_w = self.encoder_self_attention(x, x, x, average_attn_weights=False)
        x_atten_w = None
        x = x #+ x
        x = x.flatten(1)
        
        z = self.encoder(x)#.chunk(2, dim=-1)
        # z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        if kld:
            l2_loss = torch.mean(torch.abs(z))
            # kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            return z, x_atten_w
        return z, x_atten_w
    
    def decode(self, z):
        # z = self.out_norm(z)
        x= self.decoder(z)
        
        x = x.chunk(len(self.tiny_decoders), dim=-1)
        # x = torch.stack(x, dim=1)
        # x = x_atten + x
        # x = [x[:,i,:] for i in range(x.shape[1])]
        parts = []
        for part, tiny_decoder in zip(x, self.tiny_decoders):
            parts.append(tiny_decoder(part))
        return torch.cat(parts, dim=-1)

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
        # x = self.in_proj(x)
        x, x_atten_w = self.encode(x)
        x_skip = x
        skips = []
        for down in (self.downs):
            x = down(x, ada_emb, self.conditioning)
            skips.append(x)
            # raise ValueError("Not implemented, x.shape", x.shape)
        # x_weights = self.hyper_net_gen(self.conditioning)
        # x_down, x_up = x_weights.view(-1, 2048, 32, 2).chunk(2, dim=-1)
        # x_down, x_up = x_down.squeeze(-1), x_up.squeeze(-1)
     
        # x = torch.einsum('bi,bij->bj', x, x_down)
        # x = F.silu(x)
        # x = torch.einsum('bi,bij->bj', x, x_up.permute(0, 2, 1))
        for up, skip in zip(self.ups, reversed(skips)):
            # x = torch.cat([x, skip], dim=-1)
            x = up(x, ada_emb,  self.conditioning) + skip

        
        x = self.out_norm(torch.cat([x, x_skip], dim=-1)) 
        if face_embeddings is not None:
            pred_cond = self.reconstruct_conditioning(x)
        else:
            pred_cond = None
        # x = self.out_proj(x)
        x = self.decode(x)
        
        return x, pred_cond

class LoraLinear(torch.nn.Module):

    def __init__(self, data_dim=1_365_504, 
                      cond_dim=512,
                      model_dim=1024,
                    ):
        super().__init__()

        self.cond_dim = cond_dim

        self.cond_gen = nn.Linear(cond_dim, cond_dim)

        self.in_cond = nn.Sequential(
            # nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.mean_logvar_branch = nn.Linear(model_dim, data_dim)
        

        self.mean_cc = nn.Linear(data_dim, data_dim)
        self.logvar_cc = nn.Linear(data_dim, data_dim)
        self.cond_proj = nn.Linear(data_dim, cond_dim)
        ff_mult = 2
        num_blocks = 6
        layers_per_block = 4
        self.downs = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim, in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=cond_dim) for _ in range(num_blocks)])
        self.ups = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim,  in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=cond_dim) for _ in range(num_blocks)])


        
        

    def forward(self, face_embeddings):
        face_emb = face_embeddings.detach()
        face_emb = face_emb / face_emb.norm(dim=-1, keepdim=True)

        face_emb = self.cond_gen(face_emb)
        # mean, logvar = .chunk(2, dim=-1)
        x = self.in_cond(face_emb)
        skips = []
        for down in (self.downs):
            x = down(x, face_emb, None)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, face_emb,  None) + skip

        data = self.mean_logvar_branch(x)#.chunk(2, dim=-1)

        cond_proj = self.cond_proj(data)
        return data, cond_proj


        

# class LoraDiffusion(torch.nn.Module):

#     def __init__(self, data_dim=1_365_504, 
#                     model_dim=256, 
#                     ff_mult=3, 
#                     chunks=1, 
#                     act=torch.nn.SiLU, 
#                     num_blocks=4, 
#                     layers_per_block=3,
#                       cond_dropout_prob=0.3 ,
#                       cond_dim=512
#                     ):
#         super().__init__()
#         self.cond_dropout_prob = cond_dropout_prob
#         self.time_embed = TimestepEmbedding(model_dim//2, model_dim//2)
#         self.cond_dim = cond_dim

#         self.in_norm = nn.LayerNorm(data_dim)
#         self.in_proj = nn.Linear(data_dim, model_dim)
#         self.out_proj = nn.Linear(model_dim, data_dim)
#         self.out_norm = nn.LayerNorm(model_dim)

#         # self.out_proj.weight.data = self.in_proj.weight.data.T


#         self.cond_learn = nn.Linear(cond_dim , cond_dim)
#         self.cond_norm = nn.LayerNorm(cond_dim)

#         self.cond_out_proj = nn.Sequential(
#             nn.LayerNorm(model_dim),
#             nn.Linear(model_dim, cond_dim),
#         )

#         self.conditioning = None

#         self.downs = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim, in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])
#         self.ups = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, cond_dim=cond_dim,  in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])

#     def get_conditioning(self, face_embeddings):
#         if face_embeddings is not None:
#             face_emb = face_embeddings.detach()
#             # _, face_emb = face_emb.chunk(2, dim=-1)
#             # print("face_emb", face_emb.shape)
#             # full_emb = F.normalize(full_emb, p=2, dim=-1)
#             # face_emb = F.normalize(face_emb, p=2, dim=-1)

#             # face_emb = torch.cat([full_emb, face_emb], dim=-1)
#             # face_emb = self.cond_learn(face_emb)
#             # face_emb = self.cond_norm(face_emb)
#             self.conditioning = face_emb
#         else:
#             self.conditioning = None


#     def reconstruct_conditioning(self, x):
#         # pred_cond = self.cond_out(x)
#         embedding = self.cond_out_proj(x)

#         return embedding



#     def forward(self, x, t, face_embeddings=None):
#         ada_emb = self.time_embed(t)
        
#         self.get_conditioning(face_embeddings)
#         # lora_dim = 10_000
#         x = self.in_norm(x)
#         x = self.in_proj(x)
#         # x_skip = x
#         skips = []
#         for down in (self.downs):
      
#             x = down(x, ada_emb, self.conditioning)
#             skips.append(x)
#         for up, skip in zip(self.ups, reversed(skips)):
      
#             x = up(x, ada_emb,  self.conditioning) + skip


#         if face_embeddings is not None:
#             pred_cond = self.reconstruct_conditioning(x)
#         else:
#             pred_cond = None

#         x = self.out_norm(x) #+ x_skip
#         x = self.out_proj(x)
        
#         return x, pred_cond


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