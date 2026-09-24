# UNITE-AUDIO

Official project page: https://runwushi.github.io/Unite-Audio/

This repository contains the UNITE-AUDIO inference runtime for CUDA and Apple
Silicon (MPS), together with the project website in `docs/`.

Model weights will be released separately. The release will provide two
official inference settings:

- `default`: S3 Flow EMA + Base Decoder, used for the paper's main results.
- `spectral-stable`: S3 Flow EMA + Spectral Decoder, for more stable
  high-frequency detail.

See `inference/` for the runtime and platform-specific setup instructions.
