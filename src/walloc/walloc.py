import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.models.autoencoders as autoencoders
import einops
import numpy as np
from types import SimpleNamespace
from einops import rearrange
from pytorch_wavelets import DWTForward, DWT1DForward, DWTInverse, DWT1DInverse
from torch.distributions import Normal
from torchvision.transforms import ToPILImage, PILToTensor
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse

def load_config(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return SimpleNamespace(**data)

def save_config(config, path):
    with open(path, 'w') as file:
        json.dump(vars(config), file, indent=4)

class ToUniform(nn.Module):
    def __init__(self,scale,latent_max):
        super(ToUniform, self).__init__()
        self.scale = scale
        self.latent_max = latent_max
        self.normal = Normal(loc=0,scale=1)
    def forward(self, x):
        x = x/self.scale
        x = self.normal.cdf(x)
        x = x - 0.5
        x = 2*self.latent_max*x
        return x

class Round(nn.Module):
    def __init__(self):
        super(Round, self).__init__()
    def forward(self, x):
        if self.training:
            noise = torch.rand_like(x) - 0.5
            return x + noise
        else:
            return torch.round(x)

class ToNormal(nn.Module):
    def __init__(self,scale,latent_max):
        super(ToNormal, self).__init__()
        self.scale = scale
        self.latent_max = latent_max
        self.normal = Normal(loc=0,scale=1)
    def forward(self, x):
        x = x/(2*self.latent_max)
        x = x + 0.5
        x = nn.Hardtanh(min_val=0.0001, max_val=0.9999)(x)
        x = self.normal.icdf(x)
        x = self.scale*x
        return x
        
class Codec2D(nn.Module):
    def __init__(self, channels, J, Ne, Nd, latent_dim, latent_bits, lightweight_encode):
        super().__init__()
        self.channels = channels
        self.J = J
        self.freq_bands = 4**J
        self.Ne = Ne
        self.Nd = Nd
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.latent_max = 2**(latent_bits-1)-1+0.5-1e-3
        self.wt  = DWTForward(J=1, mode='periodization', wave='bior4.4')
        self.iwt = DWTInverse(mode='periodization', wave='bior4.4')
        self.clamp = torch.nn.Hardtanh(min_val=-0.5, max_val=0.5)

        entropy_bottleneck = [
            ToUniform(
                scale = (2**(latent_bits-1)-1)/1.85,
                latent_max = self.latent_max
            ),
            Round(), 
        ]

        if lightweight_encode:
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels * self.freq_bands,
                    out_channels=self.latent_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                ),
                *entropy_bottleneck
            )
        else:
            self.encoder = nn.Sequential(
                autoencoders.autoencoder_kl.Encoder(
                    in_channels=self.channels * self.freq_bands,
                    out_channels=self.latent_dim,
                    down_block_types=('DownEncoderBlock2D',),
                    block_out_channels=(Ne,),
                    layers_per_block=2,
                    norm_num_groups=32,
                    act_fn='silu',
                    double_z=False,
                    mid_block_add_attention=True,
                ),
                *entropy_bottleneck
            )
            
        self.decoder = nn.Sequential(
            ToNormal(
                scale = (2**(latent_bits-1)-1)/1.85,
                latent_max = self.latent_max
            ),
            autoencoders.autoencoder_kl.Decoder(
                in_channels = self.latent_dim,
                out_channels = self.channels*self.freq_bands,
                up_block_types = ('UpDecoderBlock2D',),
                block_out_channels = (Nd,),
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
        tf_loss = F.mse_loss(X, X_hat)
        return self.clamp(x_hat), F.mse_loss(x,x_hat), tf_loss

class Codec1D(nn.Module):
    def __init__(self, channels, J, Ne, Nd, latent_dim, latent_bits, lightweight_encode, post_filter):
        super().__init__()
        self.channels = channels
        self.J = J
        self.freq_bands = 2**J
        self.Ne = Ne
        self.Nd = Nd
        self.lightweight_encode = lightweight_encode
        self.post_filter = post_filter
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.latent_max = 2**(latent_bits-1)-1+0.5-1e-3
        self.wt  = DWT1DForward(J=1, mode='periodization', wave='bior4.4')
        self.iwt = DWT1DInverse(mode='periodization', wave='bior4.4')
        self.clamp = torch.nn.Hardtanh(min_val=-0.5, max_val=0.5)

        entropy_bottleneck = [
            ToUniform(
                scale = (2**(latent_bits-1)-1)/1.85,
                latent_max = self.latent_max
            ),
            Round(),
        ]

        if lightweight_encode:
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.channels * self.freq_bands,
                    out_channels=self.latent_dim,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                ),
                *entropy_bottleneck
            )
        else:
            self.encoder = nn.Sequential(
                autoencoders.autoencoder_oobleck.OobleckEncoder(
                    encoder_hidden_size=self.latent_dim,
                    audio_channels=self.channels*self.freq_bands,
                    downsampling_ratios=2*[1],
                    channel_multiples=2*[self.Ne//self.latent_dim],
                ),
                *entropy_bottleneck
            )
            
        self.decoder = nn.Sequential(
            ToNormal(
                scale = (2**(latent_bits-1)-1)/1.85,
                latent_max = self.latent_max
            ),
            autoencoders.autoencoder_oobleck.OobleckDecoder(
                channels=self.latent_dim,
                input_channels=self.latent_dim,
                audio_channels=self.channels*self.freq_bands,
                upsampling_ratios=2*[1],
                channel_multiples=2*[self.Nd//self.latent_dim],
            )
        )

        if post_filter:
            self.post = nn.Conv1d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=129,
                    stride=1,
                    padding=64,
                )
    
    def analysis_one_level(self,x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2),H[0].unsqueeze(2)],dim=2)
        X = einops.rearrange(X, 'b c f ℓ -> b (c f) ℓ')
        return X
    
    def wavelet_analysis(self,x,J=3):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x
    
    def synthesis_one_level(self,X):
        X = einops.rearrange(X, 'b (c f) ℓ -> b c f ℓ', f=2)
        L, H = torch.split(X, [1, 1], dim=2)
        L = L.squeeze(2)
        H = [H.squeeze(2)]
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
        if self.post_filter:
            x_hat = self.post(x_hat)
        tf_loss = F.mse_loss( X, X_hat )
        return self.clamp(x_hat), F.mse_loss(x,x_hat), tf_loss

class LinearCodec2D(nn.Module):
    def __init__(self, channels, J, latent_dim, latent_bits):
        super().__init__()
        self.channels = channels
        self.J = J
        self.freq_bands = 4**J
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.latent_max = 2**(latent_bits-1)-1+0.5-1e-3
        self.wt  = DWTForward(J=1, mode='periodization', wave='bior4.4')
        self.iwt = DWTInverse(mode='periodization', wave='bior4.4')
        self.clamp = torch.nn.Hardtanh(min_val=-0.5, max_val=0.5)
        entropy_bottleneck = [
            ToUniform(
                scale = (2**(latent_bits-1)-1)/1.85,
                latent_max = self.latent_max
            ),
            Round(),
        ]
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels * self.freq_bands,
                out_channels=self.latent_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            *entropy_bottleneck
        )
        self.decoder = nn.Sequential(
            ToNormal(
                scale = (2**(latent_bits-1)-1)/1.85,
                latent_max = self.latent_max
            ),
            nn.Conv2d(
                in_channels=self.latent_dim,
                out_channels=self.channels * self.freq_bands,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
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
        tf_loss = F.mse_loss(X, X_hat)
        return self.clamp(x_hat), F.mse_loss(x,x_hat), tf_loss

class ResidualCodec2D(torch.nn.Module):
    def __init__(self, channels, J, latent_dim, latent_bits, num_stages):
        super(ResidualCodec2D, self).__init__()
        self.channels = channels
        self.J = J
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.num_stages = num_stages
        self.latent_max = 2 ** (latent_bits - 1) - 1 + 0.5 - 1e-3
        self.wt = DWTForward(J=1, mode="periodization", wave="bior4.4")
        self.iwt = DWTInverse(mode="periodization", wave="bior4.4")
        self.clamp = torch.nn.Hardtanh(min_val=-0.5, max_val=0.5)

        # Entropy bottleneck
        self.to_uniform = ToUniform(
            scale=(2 ** (latent_bits - 1) - 1) / 1.85, latent_max=self.latent_max
        )
        self.round = Round()
        self.to_normal = ToNormal(
            scale=(2 ** (latent_bits - 1) - 1) / 1.85, latent_max=self.latent_max
        )

        # Linear encoders and decoders
        self.encoders = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels=self.channels * (4 ** J),
                out_channels=self.latent_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for _ in range(num_stages)
        ])
        self.decoders = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels=self.latent_dim,
                out_channels=self.channels * (4 ** J),
                kernel_size=1,
                stride=1,
                padding=0,
            ) for _ in range(num_stages)
        ])

    def analysis_one_level(self, x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2), H[0]], dim=2)
        X = einops.rearrange(X, "b c f h w -> b (c f) h w")
        return X

    def wavelet_analysis(self, x, J=3):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x

    def synthesis_one_level(self, X):
        X = einops.rearrange(X, "b (c f) h w -> b c f h w", f=4)
        L, H = torch.split(X, [1, 3], dim=2)
        L = L.squeeze(2)
        H = [H]
        y = self.iwt((L, H))
        return y

    def wavelet_synthesis(self, x, J=3):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x

    def forward(self, x):
        X = self.wavelet_analysis(x, J=self.J)
        residual = X
        total_loss = 0
        tf_losses = []
        recon_losses = []

        cumulative_reconstruction = torch.zeros_like(X)
        for i_stage in range(self.num_stages):
            # Encode
            Z = self.encoders[i_stage](residual)
            Z = self.to_uniform(Z)
            Z = self.round(Z)

            # Decode
            Z = self.to_normal(Z)
            X_hat = self.decoders[i_stage](Z)

            # Accumulate loss
            tf_loss = torch.nn.functional.mse_loss(residual,X_hat)
            recon_loss = torch.nn.functional.mse_loss(
                self.wavelet_synthesis(residual, J=self.J),
                self.wavelet_synthesis(X_hat, J=self.J)
            )
            tf_losses.append(tf_loss)
            recon_losses.append(recon_loss)
            total_loss += recon_loss

            # Update cumulative reconstruction
            cumulative_reconstruction = cumulative_reconstruction + X_hat
            residual = residual - X_hat

        x_hat = self.wavelet_synthesis(cumulative_reconstruction, J=self.J)
        return self.clamp(x_hat), total_loss, tf_losses, recon_losses

    def rae_encode(self, x):
        X = self.wavelet_analysis(x, J=self.J)
        residual = X
        total_loss = 0
        tf_losses = []
        recon_losses = []
        cumulative_reconstruction = torch.zeros_like(X)
        latent = []
        for i_stage in range(self.num_stages):
            Z = self.encoders[i_stage](residual)
            Z = self.to_uniform(Z)
            Z = self.round(Z)
            latent.append(Z)
            Z = self.to_normal(Z)
            X_hat = self.decoders[i_stage](Z)
            tf_loss = torch.nn.functional.mse_loss(residual,X_hat)
            recon_loss = torch.nn.functional.mse_loss(
                self.wavelet_synthesis(residual, J=self.J),
                self.wavelet_synthesis(X_hat, J=self.J)
            )
            tf_losses.append(tf_loss)
            recon_losses.append(recon_loss)
            total_loss += recon_loss
            cumulative_reconstruction = cumulative_reconstruction + X_hat
            residual = residual - X_hat
        x_hat = self.wavelet_synthesis(cumulative_reconstruction, J=self.J)
        return self.clamp(x_hat), torch.cat(latent,dim=1)
        
    def rae_decode(self, latent):
        latent_splits = torch.split(latent, self.latent_dim, dim=1)
        cumulative_reconstruction = torch.zeros(
            (latent.shape[0], self.channels*(4**self.J), *latent.shape[2:]), 
            device=latent.device
        )
        for i_stage in range(self.num_stages):
            Z = latent_splits[i_stage]
            Z = self.to_normal(Z)
            X_hat = self.decoders[i_stage](Z)
            cumulative_reconstruction = cumulative_reconstruction + X_hat
        x_hat = self.wavelet_synthesis(cumulative_reconstruction, J=self.J)
        return self.clamp(x_hat)

def to_bytes(x, n_bits):
    max_value = 2**(n_bits - 1) - 1
    min_value = -max_value - 1
    if x.min() < min_value or x.max() > max_value:
        raise ValueError(f"Tensor values should be in the range [{min_value}, {max_value}].")
    return (x + (max_value + 1)).to(torch.uint8)

def from_bytes(x, n_bits):
    max_value = 2**(n_bits - 1) - 1
    return (x.to(torch.float32) - (max_value + 1))

def concatenate_channels(x, C):
    batch_size, N, h, w = x.shape
    if N % C != 0 or int((N // C)**0.5) ** 2 * C != N:
        raise ValueError(f"Number of channels must satisfy N = {C} * (n^2) for some integer n.")
    
    n = int((N // C)**0.5)
    x = rearrange(x, 'b (c nh nw) h w -> b (nh h) (nw w) c', c=C, nh=n, nw=n)
    return x

def split_channels(x, N, C):
    batch_size, _, H, W = x.shape
    n = int((N // C)**0.5)
    h = H // n
    w = W // n
    
    x = rearrange(x, 'b c (nh h) (nw w) -> b (c nh nw) h w', c=C, nh=n, nw=n)
    return x

def latent_to_pil(latent, n_bits, C):
    latent_bytes = to_bytes(latent, n_bits)
    concatenated_latent = concatenate_channels(latent_bytes, C)
    
    if C == 1:
        mode = 'L'
        concatenated_latent = concatenated_latent.squeeze(-1)
    elif C == 3:
        mode = 'RGB'
    elif C == 4:
        mode = 'CMYK'
    else:
        raise ValueError(
            f"Unsupported number of channels C={C}. Supported values are 1 (L), 3 (RGB), and 4 (CMYK)"
        )
    
    pil_images = []
    for i in range(concatenated_latent.shape[0]):
        pil_image = Image.fromarray(concatenated_latent[i].numpy(), mode=mode)
        pil_images.append(pil_image)
    
    return pil_images

def pil_to_latent(pil_images, N, n_bits, C):
    tensor_images = [PILToTensor()(img).unsqueeze(0) for img in pil_images]
    tensor_images = torch.cat(tensor_images, dim=0)
    split_latent = split_channels(tensor_images, N, C)
    latent = from_bytes(split_latent, n_bits)
    return latent

def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div
    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )
    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top
    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)
    return pad, unpad

def pad(x, p=16, mode='reflect'):
    h, w = x.size(2), x.size(3)
    pad, _ = compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode=mode)

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    _, unpad = compute_padding(h, w, out_h=H, out_w=W)
    return F.pad(x, unpad)