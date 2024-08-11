import torch
from torch import nn


import sys
sys.path.append('..')

from common.models import ChunkFanOut, DiTBlockNoAda, AttentionResampler, Resnet
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, in_proj_chunks=1, act=torch.nn.SiLU, num_layers=6, latent_dim=None):
        super().__init__()
        self.in_norm = nn.LayerNorm(data_dim)
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        # latent_dim = model_dim * 2 if latent_dim is None else model_dim
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim*2)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_norm(x)
        return self.out_proj(x).chunk(2, dim=-1)
        # return mean, logvar


class Decoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, out_proj_chunks=1, act=torch.nn.SiLU, num_layers=6,  latent_dim=None):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        self.out_proj = ChunkFanOut(model_dim, data_dim, chunks=out_proj_chunks)
        self.out_norm = nn.LayerNorm(model_dim)
        # self.out_norm_2 = nn.LayerNorm(data_dim) # this is a nice way to get full size parameters while still fairly cheap
        

    def forward(self, x):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x

class BiasAdd(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias


class SparseLinear(nn.Module):
    
    def __init__(self, full_in_dim=1024, full_out_dim=1024, heads=8, bias=True):
        super(SparseLinear, self).__init__()
        self.full_in = full_in_dim
        self.full_out = full_out_dim
        self.in_dim = full_in_dim // heads
        self.out_dim = full_out_dim // heads
        self.h = heads
        weights = [torch.randn(self.in_dim, self.out_dim) for _ in range(heads)]
        for i in range(len(weights)):
            torch.nn.init.xavier_uniform_(weights[i])
        self.weight = nn.Parameter(torch.stack(weights, dim=0))
        self.bias_add = BiasAdd(self.full_out) if bias else nn.Identity()

    def forward(self, x):
        b, h, in_dim = x.shape[0], self.h, self.in_dim
        x = x.reshape(b, h, in_dim)
        x = torch.einsum('bhd,hdl->bhl', x, self.weight)
        x = x.reshape(b, h * self.out_dim)
        x = self.bias_add(x)
        return x

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

class LoraVAE(torch.nn.Module):

    def __init__(self, input_dim=99_648, latent_dim=2048):
        super(LoraVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        tiny_encoders = []
        tiny_decoders = [] # transpose of encoder
        for i, (_, _, _, dim, _) in enumerate(config):
            tiny_encoder = nn.Linear(dim, 320)
            tiny_decoder = nn.Linear(320, dim)
            tiny_decoder.weight.data = tiny_encoder.weight.data.T
            tiny_encoders.append(tiny_encoder)
            tiny_decoders.append(tiny_decoder)

        self.tiny_encoders = nn.ModuleList(tiny_encoders)
        self.tiny_decoders = nn.ModuleList(tiny_decoders)

        self.encoder_self_attention = nn.MultiheadAttention(320, 8, batch_first=True)

        # Encoder
        self.encoder = nn.Sequential(
            SparseLinear(320*len(tiny_encoders), 128*len(tiny_encoders), heads=32),
            nn.SiLU(),
            nn.Linear(128*len(tiny_encoders), latent_dim*2)
        )

        # Decoder       
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*len(tiny_encoders)),
            nn.SiLU(),
            SparseLinear(128*len(tiny_encoders), 320*len(tiny_encoders), heads=32),
        )
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim),
        # )

        # # Decoder       
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, input_dim),
        # )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def encode(self, x, kld=False, mean_logvar=False, actual_noise=None):
        parts = []
        for (_, start, end, _, _), tiny_encoder in zip(config, self.tiny_encoders):
            part = tiny_encoder(x[:, start:end])
            parts.append(part)
        x = torch.stack(parts, dim=1)
        x_atten, x_atten_w = self.encoder_self_attention(x, x, x, average_attn_weights=False)
        x = x_atten + x
        x = x.flatten(1)
        if mean_logvar:
            return self.encoder(x)
        mean,logvar = self.encoder(x).chunk(2, dim=-1)
        if (actual_noise is None):
            actual_noise = torch.randn_like(mean)
        z = mean + actual_noise * torch.exp(0.5 * logvar)
        if kld:
            l2_loss = torch.mean(torch.abs(z))
            # kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            return z, l2_loss, x_atten_w
        return z, x_atten_w

    def standardized_z(self, z):
        # z = (z - latent_mean) / latent_std
        return z #/ 0.8
    def unstandardized_z(self, z):
        # z = z * latent_std + latent_mean
        return z #* 0.8

    def decode(self, z):
        x= self.decoder(z)
        
        x = x.chunk(len(self.tiny_decoders), dim=-1)
        # x = torch.stack(x, dim=1)
        # x = x_atten + x
        # x = [x[:,i,:] for i in range(x.shape[1])]
        parts = []
        for part, tiny_decoder in zip(x, self.tiny_decoders):
            parts.append(tiny_decoder(part))
        return torch.cat(parts, dim=-1)
    def apply_std_on_weights(self, weights):
        weight_parts = []
        for (key, start, end, length, std) in config:
            weight_parts.append(weights[:, start:end] / std)
        return torch.cat(weight_parts, dim=-1) #/ 1.2
    def deapply_std_on_weights(self, weights):
        weight_parts = []
        for (key, start, end, length, std) in config:
            weight_parts.append((weights[:, start:end]) * std)
        return torch.cat(weight_parts, dim=-1)
    def split(self, x):
        parts = []
        for (_, start, end, _, _) in config:
            parts.append(x[:, start:end])
        return parts
    def forward(self, x):
        z, kld, enc_atten = self.encode(x, True)
        pred = self.decode(z)
        return pred, z, kld, (enc_atten,)



class BiasAdd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias

class SparseAE(nn.Module):
    def __init__(self, expansion_factor=10, l0_alpha=1e-2):
        super(SparseAE, self).__init__()
        latent_dim = 512 
        self.l0_alpha = l0_alpha
        expanded_dim = round(latent_dim * expansion_factor)

        self.dictionary_size = expanded_dim
        self.up = nn.Linear(latent_dim, expanded_dim, bias=False)
        self.down = nn.Linear(expanded_dim, latent_dim, bias=False)

        # self.precond = nn.LayerNorm(latent_dim)
        # self.mid_cond = nn.LayerNorm(round(expanded_dim))
        # self.out_cond = nn.LayerNorm(latent_dim)
        self.apply(self._init_weights)

    def initiate_vae(self,vae_weights_path,  **vae_params): # initiate the base VAE, make sure same weight is used during training and inference of this SAE
        self.vae = LoraVAE(**vae_params)
        self.vae.load_state_dict(torch.load(vae_weights_path))
        self.freeze_vae()
       

    def normalize_dictionary(self):
        self.down.weight.data = F.normalize(self.down.weight.data, p=2, dim=0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is self.down:
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu') 
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        
        if isinstance(module, BiasAdd):
            module.bias.data.fill_(0.01)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.fill_(0.01)
            module.weight.data.fill_(0.01)

        self.down.weight.data = self.up.weight.data.T
    
    def freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    def corr_loss(self, encoded_relu):
        encoded_relu = encoded_relu - encoded_relu.mean(dim=0)
        encoded_relu = encoded_relu / (encoded_relu.norm(dim=0) + 1e-8)
        corr = encoded_relu.T @ encoded_relu
        target = torch.eye(corr.shape[0], device=corr.device)
        return F.mse_loss(corr, target)
    
    def forward(self, x):
        with torch.no_grad():
            mu,logvar = self.vae.encoder(x)
        
        # mu = self.precond(_mu)
        dense_dictionary = self.up(mu)
        dictionary = F.relu(dense_dictionary)
        # self.normalize_dictionary()
        # decoder_input = self.mid_cond(dictionary)
        recon_x = self.down(dictionary)
        # recon_x = self.out_cond(recon_x)

        eps = 1e-3
        
        l0_loss = ((F.sigmoid((dense_dictionary - eps) * 1000.0) ).sum() / self.dictionary_size) * self.l0_alpha  # approximated l0 loss
        corr_loss = self.corr_loss(dictionary) * 0.2
            
        return recon_x, mu, dictionary, l0_loss, corr_loss





class Discriminator(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, 
                        model_dim=256, 
                        ff_mult=3, 
                        in_proj_chunks=1, 
                        act=torch.nn.SiLU, 
                        num_layers=6, 
                        ):
        super().__init__()
        self.in_norm = nn.LayerNorm(data_dim)
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        latent_dim = model_dim * 2 if latent_dim is None else model_dim
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_norm(x)
        return self.out_proj(x)