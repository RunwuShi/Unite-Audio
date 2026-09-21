# Unite-Audio inference

This folder is the standalone text-to-audio runtime. It references the internal
bundle's sibling `../checkpoints/` directory and contains only inference files.

## Install

Use Python 3.10+ and install a CUDA-compatible `torch` and `torchaudio` build
first, following [PyTorch's install matrix](https://pytorch.org/get-started/locally/).
The release environment used torch/torchaudio 2.4.0 with CUDA 12.4.  Then run:

```bash
cd unite-audio_final/inference
pip install -r requirements.txt
```

The frozen text encoder is not packaged with these checkpoints.  The default
`google/flan-t5-large` downloads through Hugging Face on first use.  To run
offline, provide a local model directory and `--local-files-only`.

## Generate

The default is the RL14 checkpoint. This command writes float32 mono
16-kHz WAV files plus `metadata.json` and `run.json`.

```bash
cd unite-audio_final/inference
python generate.py --caption "Rain falls on a quiet city street" --device cuda
```

Choose a bundled model or an explicit checkpoint:

```bash
python generate.py --model decoder_mid3_gan --caption "A soft piano melody" --device cuda
python generate.py --checkpoint ../checkpoints/s3_rl14 --caption "Ocean waves" --device cuda
python generate.py --model s3_rl14 --text-encoder /models/flan-t5-large \
  --local-files-only --caption "A train passes by" --device cuda
```

Each model directory contains the full resumable training checkpoint. Inference
loads its model state directly: S1/S2/S3 default to EMA; the decoder variants
default to online weights. Override this with `--weights ema` or `--weights
online`. The defaults use 10 s, Euler, 32 prior steps, one decoder step, seed
1234, CFG 4 for the three stage checkpoints and CFG 5 for decoder variants.

## Included source

`task_audio/` contains only the fixed runtime needed by the four retained
checkpoints: prompt conditioning, latent sampling, waveform decoding, and
checkpoint loading. It is code only; all learned parameters are in the bundled
checkpoints. Dataset, training, evaluation, and auxiliary model code are not
included.
