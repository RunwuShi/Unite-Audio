# Internal training workspace

`train.py` preserves the actual S1 → S2 → S3/RL14 → decoder training path.
The full resumable checkpoints remain in the CPFS internal archive; this Git
package includes only selected inference states. It does not create
or invoke periodic ESC50, AudioCaps, or 886 evaluation jobs. The top-level
`../evaluation/` manifest is the sole 886 evaluation contract.

The training source has three top-level packages only: `task_audio`,
`task_audio_pmf`, and `task_audio_preference_rl`. Shared model internals live
under the private `task_audio._backbone` namespace.

Install the environment, then copy and edit the local path file:

```bash
cd unite-audio_final/train
pip install -r requirements.txt
cp data_paths.example.yaml data_paths.yaml
```

Run pMF stages with the original process count (the formal S1/S2 and decoder
recipes used eight GPUs):

```bash
PYTHONPATH=src accelerate launch --num_processes 8 train.py --stage s2_noisy_30ep
PYTHONPATH=src accelerate launch --num_processes 8 train.py --stage decoder_spectral
```

S3/RL14 keeps its formal two-GPU GRPO constraint and its reward/cache setup:

```bash
PYTHONPATH=src accelerate launch --num_processes 2 train.py --stage s3_rl14
```

Use `--resume-from ../checkpoints/<id>` to continue an archived run exactly.
S2 initializes from the archived S1 checkpoint; decoder recipes initialize from
the archived S3 checkpoint. The historical S1 predecessor is not bundled, so a
fresh S1 start needs an external base checkpoint; its bundled checkpoint can
still be resumed directly. `decoder_mid3_gan` retains the selected full-scope
GAN checkpoint; a fresh run starts from S3, while exact continuation uses its
own checkpoint.

All resolved paths and overrides are written to `<output>/config/resolved_config.json`.
