import torch
import torch.nn as nn
import einops
from pytorch_wavelets import DWTForward, DWTInverse
import diffusers.models.autoencoders as autoencoders 

class Round(nn.Module):
    def __init__(self):
        super(Round, self).__init__()
    def forward(self, x):
        if self.training:
            noise = torch.rand_like(x) - 0.5
            return x + noise
        else:
            return torch.round(x)
        
class Walloc(nn.Module):
    def __init__(self, channels, J, N, latent_dim, latent_bits):
        super().__init__()
        self.channels = channels
        self.J = J
        self.freq_bands = 4**J
        self.N = N
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.latent_max = clamp_value = 2 ** (latent_bits - 1) - 0.501
        self.wt  = DWTForward(J=1, mode='periodization', wave='bior4.4')
        self.iwt = DWTInverse(mode='periodization', wave='bior4.4')
        self.clamp = torch.nn.Hardtanh(min_val=-0.5, max_val=0.5)
        self.encoder = nn.Sequential(
            autoencoders.autoencoder_kl.Encoder(
                in_channels = self.channels*self.freq_bands,
                out_channels = self.latent_dim,
                down_block_types = ('DownEncoderBlock2D',),
                block_out_channels = (N,),
                layers_per_block = 2,
                norm_num_groups = 32,
                act_fn = 'silu',
                double_z = False,
                mid_block_add_attention=True,
            ),
            torch.nn.Hardtanh(min_val= -self.latent_max, max_val=self.latent_max),
            Round()
        )
        self.decoder = nn.Sequential(
                autoencoders.autoencoder_kl.Decoder(
                    in_channels = self.latent_dim,
                    out_channels = self.channels*self.freq_bands,
                    up_block_types = ('UpDecoderBlock2D',),
                    block_out_channels = (N,),
                    layers_per_block = 2,
                    norm_num_groups = 32,
                    act_fn = 'silu',
                    mid_block_add_attention=True,
                ),
            )
        
    def analysis_one_level(self,x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2),H[0]],dim=2)
        X = einops.rearrange(X, 'b c f h w -> b (c f) h w')
        return X
    
    def wavelet_analysis(self,x,J=3):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x
    
    def synthesis_one_level(self,X):
        X = einops.rearrange(X, 'b (c f) h w -> b c f h w', f=4)
        L, H = torch.split(X, [1, 3], dim=2)
        L = L.squeeze(2)
        H = [H]
        y = self.iwt((L, H))
        return y
    
    def wavelet_synthesis(self,x,J=3):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x
            
    def forward(self, x):
        X = self.wavelet_analysis(x,J=self.J)
        Y = self.encoder(X)
        X_hat = self.decoder(Y)
        x_hat = self.wavelet_synthesis(X_hat,J=self.J)
        tf_loss = F.mse_loss( X, X_hat )
        return self.clamp(x_hat), F.mse_loss(x,x_hat), tf_loss