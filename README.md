# **UNITE-AUDIO**: Joint learning of continuous tokenization and latent flow matching for text-to-audio generation

<p align="center">
  <a href="https://runwushi.github.io/Unite-Audio/"><img src="https://img.shields.io/badge/Project%20Page-202421?style=flat&logo=googlechrome&logoColor=white" alt="Project Page" /></a>
  <a href="https://arxiv.org/abs/2609.28206"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat&logo=arxiv&logoColor=white" alt="arXiv" /></a>
  <img src="https://img.shields.io/badge/Hugging%20Face-coming%20soon-FFB000?style=flat&logo=huggingface&logoColor=white" alt="Hugging Face coming soon" />
</p>

<table>
  <tr>
    <td align="center"><img src="docs/assets/method-overview.png" height="280" alt="Separate two-stage training versus Unite-Audio joint training" /><br /><sub>Separate two-stage training versus our joint single-stage training.</sub></td>
    <td align="center"><img src="docs/assets/method-details.png" height="280" alt="UNITE-AUDIO reconstruction and generative training paths" /><br /><sub>Detailed reconstruction and generative training paths in UNITE-AUDIO.</sub></td>
  </tr>
</table>

## Checkpoints

| Checkpoint | Description |
| --- | --- |
| Stage 3 Flow Model | The shared text-conditioned latent flow model. |
| Default Decoder | The decoder checkpoint used for the reported paper metrics. |
| Spectral Decoder | An alternative decoder with more stable high-frequency detail. |

The default setting uses the Stage 3 Flow Model with the Spectral Decoder.

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
