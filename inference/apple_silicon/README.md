# Apple Silicon inference

This launcher runs UNITE-AUDIO on the Apple GPU through MPS.

## Setup

```bash
cd inference/apple_silicon
./setup.sh
```

The first generation downloads `google/flan-t5-large` from Hugging Face. The
UNITE-AUDIO weights are released separately and must be placed in the
repository's `checkpoints/` directory.

## Generate

The default setting is the paper configuration: S3 Flow EMA with the Base
Decoder. It generates 10 seconds of audio with 32 prior steps:

```bash
python run.py "Ocean waves crash against a rocky shore" 32
```

To use the Spectral Decoder, edit `inference_config.json` and set
`"decoder_model": "spectral"`. This optional setting is intended for more
stable high-frequency detail.

If an MPS operation is unsupported, PyTorch CPU fallback is enabled.
