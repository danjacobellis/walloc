# Wavelet Learned Lossy Compression

- [WaLLoC repository](https://github.com/danjacobellis/walloc/)
- [Paper: "Learned Compression for Compressed Learning"](https://danjacobellis.net/_static/walloc.pdf)
- [Additional code accompanying the paper](https://github.com/danjacobellis/lccl)
- [Pre-trained codecs available on Hugging Face](https://huggingface.co/danjacobellis/walloc)


```{figure} img/radar.svg
---
name: radar
---
Comparison of WaLLoC with other autoencoder designs for RGB Images and stereo audio.

```

```{figure} img/wpt.svg
---
name: walloc
---
Example of forward and inverse WPT with $J=2$ levels. Each level applies filters $\text{L}_{\text{A}}$ and $\text{H}_{\text{A}}$ independently to each of the signal channels, followed by downsampling by two $(\downarrow 2)$. An inverse level consists of upsampling $(\uparrow 2)$ followed by $\text{L}_{\text{S}}$
and $\text{H}_{\text{S}}$, then summing the two channels. The full WPT $\tilde{\textbf{X}}$ of consists of $J$ levels.

```

```{figure} img/walloc.svg
---
name: walloc
---
WaLLoC’s encode-decode pipeline. The entropy bottleneck and entropy coding steps are only required to achieve high compression ratios for storage and transmission. For compressed-domain learning where dimensionality reduction is the primary goal, these steps can be skipped to reduce overhead and completely eliminate CPU-GPU transfers.

```