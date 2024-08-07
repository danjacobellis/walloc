---
datasets:
- danjacobellis/LSDIR_540
---
# Wavelet Learned Lossy Compression (WaLLoC)

## Installation

1. Follow the installation instructions for [torch](https://pytorch.org/get-started/locally/)
2. Install WaLLoC and other dependencies via pip

```pip install walloc PyWavelets pytorch-wavelets```

##  Pre-trained checkpoints

Pre-trained checkpoints are available on [Hugging Face](https://huggingface.co/danjacobellis/walloc)

## Usage example


```python
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display
from torchvision.transforms import ToPILImage, PILToTensor
from walloc.walloc import Walloc
class Args: pass
```

### Load the model from a pre-trained checkpoint

```wget https://hf.co/danjacobellis/walloc/resolve/main/v0.6.3_ext.pth```


```python
device = "cpu"
checkpoint = torch.load("v0.6.3_ext.pth",map_location="cpu")
args = checkpoint['args']
codec = Walloc(
    channels = args.channels,
    J = args.J,
    N = args.N,
    latent_dim = args.latent_dim,
    latent_bits = 5
)
codec.load_state_dict(checkpoint['model_state_dict'])
codec = codec.to(device)
```

### Load an example image

```wget "https://r0k.us/graphics/kodak/kodak/kodim05.png"```


```python
img = Image.open("kodim05.png")
img
```




    
![png](README_files/README_6_0.png)
    



### Full encoding and decoding pipeline with .forward()

* If `codec.eval()` is called, the latent is rounded to nearest integer.

* If `codec.train()` is called, uniform noise is added instead of rounding.


```python
with torch.no_grad():
    codec.eval()
    x = PILToTensor()(img).to(torch.float)
    x = (x/255 - 0.5).unsqueeze(0).to(device)
    x_hat, _, _ = codec(x)
ToPILImage()(x_hat[0]+0.5)
```




    
![png](README_files/README_8_0.png)
    



### Accessing latents


```python
with torch.no_grad():
    codec.eval()
    X = codec.wavelet_analysis(x,J=codec.J)
    Y = codec.encoder(X)
    X_hat = codec.decoder(Y)
    x_hat = codec.wavelet_synthesis(X_hat,J=codec.J)

print(f"dimensionality reduction: {x.numel()/Y.numel()}×")
```

    dimensionality reduction: 12.0×



```python
Y.unique()
```




    tensor([-15., -14., -13., -12., -11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,
             -3.,  -2.,  -1.,  -0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,
              9.,  10.,  11.,  12.,  13.,  14.,  15.])




```python
plt.figure(figsize=(5,3),dpi=150)
plt.hist(
    Y.flatten().numpy(),
    range=(-17.5,17.5),
    bins=35,
    density=True,
    width=0.8);
plt.title("Histogram of latents")
plt.xticks(range(-15,16,5));
```


    
![png](README_files/README_12_0.png)
    


# Lossless compression of latents using PNG


```python
grid_size = 4
n_channels, H, W = Y[0].shape
combined_image = Image.new('L', (W * grid_size, H * grid_size))
size_bytes = 0
frames = []
for i, channel in enumerate(Y[0]):
    channel = (channel+16).to(torch.uint8)
    frames.append(Image.fromarray(channel.numpy(), mode='L'))
    row = i // grid_size
    col = i % grid_size
    channel = ToPILImage()(channel)
    combined_image.paste(channel, (col * W, row * H))
combined_image
```




    
![png](README_files/README_14_0.png)
    




```python
combined_image.save('tmp.png')
print("compression_ratio: ", x.numel()/os.path.getsize("tmp.png"))
```

    compression_ratio:  20.307596963280485



```python
!jupyter nbconvert --to markdown README.ipynb
```

    [NbConvertApp] Converting notebook README.ipynb to markdown
    [NbConvertApp] Support files will be in README_files/
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Writing 12900 bytes to README.md

