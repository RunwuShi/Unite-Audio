#!/usr/bin/env python3
"""Apply narrow release-runtime compatibility fixes, then run upstream CLI."""

from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path
from typing import Mapping

from torch import Tensor, nn


def main() -> None:
    entrypoint_value = os.environ.get("UNITE_AUDIO_ENTRYPOINT")
    if not entrypoint_value:
        raise RuntimeError("UNITE_AUDIO_ENTRYPOINT is required")
    entrypoint = Path(entrypoint_value).expanduser().resolve()
    inference_root = entrypoint.parent
    sys.path.insert(0, str(inference_root))

    # Release checkpoints contain the trainable projection and null token. The
    # upstream runtime accidentally filters the full conditioner namespace.
    from task_audio import checkpoints
    from task_audio.runtime import model as runtime_model

    runtime_model._IGNORED_STATE_PREFIXES = tuple(
        prefix
        for prefix in runtime_model._IGNORED_STATE_PREFIXES
        if prefix != "text_conditioner."
    )
    flow_state = os.environ.get("UNITE_AUDIO_FLOW_STATE")
    decoder_state = os.environ.get("UNITE_AUDIO_DECODER_STATE")
    if flow_state and decoder_state:
        checkpoints.load_model_state = _mixed_state_loader(
            Path(flow_state), Path(decoder_state)
        )
    spec = importlib.util.spec_from_file_location("unite_audio_upstream_generate", entrypoint)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream entrypoint: {entrypoint}")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    # The release CLI also omitted these two imports from its generated file.
    from task_audio.infer.generation import generate_to_directory, load_captions

    upstream.generate_to_directory = generate_to_directory
    upstream.load_captions = load_captions
    upstream.main()


FLOW_PREFIXES = ("fm_encoder.", "fm_head.", "text_conditioner.")
DECODER_PREFIXES = (
    "decoder.",
    "decoder_latent_norm.",
    "latent_to_decoder.",
    "decoder_token_norm.",
    "decoder_up_blocks.",
    "decoder_upsamples.",
)


def _mixed_state_loader(flow_path: Path, decoder_path: Path):
    def load_model_state(model: nn.Module, _path: str | Path) -> tuple[list[str], list[str]]:
        from safetensors.torch import load_file

        flow: Mapping[str, Tensor] = load_file(str(flow_path), device="cpu")
        decoder: Mapping[str, Tensor] = load_file(str(decoder_path), device="cpu")
        state = {
            key: value for key, value in flow.items() if key.startswith(FLOW_PREFIXES)
        }
        state.update(
            (key, value)
            for key, value in decoder.items()
            if key.startswith(DECODER_PREFIXES)
        )
        state = model.filter_checkpoint_state(state)
        expected = model.checkpoint_model_state(to_cpu=False)
        current = model.state_dict()
        mismatched = [
            key
            for key, value in state.items()
            if key in current and tuple(value.shape) != tuple(current[key].shape)
        ]
        unexpected = sorted(set(state).difference(current))
        missing_saved = sorted(set(expected).difference(state))
        if mismatched or unexpected or missing_saved:
            raise RuntimeError(
                "mixed checkpoint is incompatible: "
                f"missing={missing_saved[:8]} unexpected={unexpected[:8]} "
                f"mismatched={mismatched[:8]}"
            )
        missing, unexpected_load = model.load_state_dict(state, strict=False)
        return list(missing), list(unexpected_load)

    return load_model_state


if __name__ == "__main__":
    main()
