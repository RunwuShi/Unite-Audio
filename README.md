# UNITE-AUDIO

Official project page: https://runwushi.github.io/Unite-Audio/

## Inference

Install the dependencies:

```bash
pip install -r requirements.txt
cd inference
```

Generate audio with the default setting:

```bash
python infer.py "Ocean waves crash against a rocky shore"
```

The default uses the Stage 3 Flow Model and Spectral Decoder. Available
devices, models, checkpoint definitions, and default sampling settings are all
listed in `config.json`. Command-line arguments override the configuration:

```bash
python infer.py "Ocean waves crash against a rocky shore" --device mps
python infer.py "Ocean waves crash against a rocky shore" --decoder default_decoder --steps 16
```

Model weights will be released separately.
