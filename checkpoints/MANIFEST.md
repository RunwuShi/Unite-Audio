# Selected inference checkpoints

| ID | Stored state | Default inference state |
| --- | --- | --- |
| `s2_noisy_30ep` | EMA | `ema_model.safetensors` |
| `s3_rl14` | EMA | `ema_model.safetensors` |
| `decoder_spectral` | online | `model.safetensors` |
| `decoder_mid3_gan` | online | `model.safetensors` |

These are inference states only. Optimizers, discriminators, RNG state, and
unselected checkpoints remain in the CPFS internal archive.
