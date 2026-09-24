# Inference

UNITE-AUDIO supports CUDA and Apple Silicon (MPS) inference. Model weights are
released separately from this repository.

Two official model settings are provided:

- `default`: S3 Flow EMA + Base Decoder. This is the paper setting.
- `spectral-stable`: S3 Flow EMA + Spectral Decoder, with more stable
  high-frequency detail.

## CUDA

Install a CUDA-compatible PyTorch and torchaudio build following the
[PyTorch installation guide](https://pytorch.org/get-started/locally/), then:

```bash
cd inference
pip install -r requirements.txt
```

## Apple Silicon

For MPS inference on Apple Silicon, see
[`apple_silicon/README.md`](apple_silicon/README.md).
