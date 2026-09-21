from __future__ import annotations

import gc
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from task_audio.training import TTATrainer


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class PMFTrainer(TTATrainer):
    """TaskAudio trainer with explicit mixture cursors and configurable eval failure handling."""

    def _should_evaluate_audiocaps(self, step: int) -> bool:
        skip_steps = {
            int(value)
            for value in self.config.audiocaps_eval_config.get("skip_steps", ())
        }
        explicit_steps = {
            int(value)
            for value in self.config.audiocaps_eval_config.get(
                "explicit_steps", ()
            )
        }
        return (
            int(step) not in skip_steps
            and (
                int(step) in explicit_steps
                or super()._should_evaluate_audiocaps(step)
            )
        )

    def train(self, train_dataloader: Any) -> int:
        self._pmf_train_dataloader = train_dataloader
        return super().train(train_dataloader)

    def _gather_rank_data_cursors(self) -> list[dict[str, Any]]:
        cursors = super()._gather_rank_data_cursors()
        loader = getattr(self, "_pmf_train_dataloader", None)
        dataset = getattr(loader, "dataset", None)
        state_fn = getattr(dataset, "state_for_offset", None)
        if callable(state_fn):
            state = state_fn(self.data_microbatch_offset)
            for cursor in cursors:
                cursor["source_range_state"] = state
        return cursors

    def save_checkpoint(self, step: int, *, last: bool = False) -> Path:
        path = super().save_checkpoint(step, last=last)
        if not last:
            return path
        # Preserve the shared checkpoint format while honoring the campaign's
        # requested number of recoverable rolling snapshots. Permanent
        # milestones keep their checkpoint-XXXXXXXX names and are not rotated.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            destination = (
                self.checkpoint_root / f"checkpoint-rolling-{int(step):08d}"
            )
            temporary = destination.with_name(destination.name + ".tmp")
            if destination.exists():
                shutil.rmtree(destination)
            if temporary.exists():
                shutil.rmtree(temporary)
            try:
                shutil.copytree(path, temporary, copy_function=os.link)
            except OSError:
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(path, temporary, copy_function=shutil.copy2)
            temporary.replace(destination)
            rolling = sorted(
                self.checkpoint_root.glob("checkpoint-rolling-*")
            )
            keep = int(self.config.keep_last_n_checkpoints)
            while keep >= 0 and len(rolling) > keep:
                shutil.rmtree(rolling.pop(0))
        self.accelerator.wait_for_everyone()
        return path

    def _evaluate_audiocaps(self, step: int) -> None:
        """Block all ranks for the full 957 eval and apply its failure policy."""

        # Explicit epoch-aligned evaluations need not coincide with the
        # periodic checkpoint cadence. Persist the exact model state before
        # launching generation so the evaluation directory cannot be paired
        # with an older rolling checkpoint.
        checkpoint = self._checkpoint_dir(step)
        if not checkpoint.is_dir():
            self.save_checkpoint(step, last=True)
            self.save_milestone(step)

        name = str(
            self.config.audiocaps_eval_config.get(
                "name", "audiocaps_meanaudio_957"
            )
        ).strip()
        if not name or "/" in name or "\\" in name:
            raise ValueError("audiocaps_eval.name must be one directory name")
        status_path = (
            self.output_dir
            / "evaluation"
            / name
            / "hook_status"
            / f"step-{step:08d}.json"
        )
        if self.accelerator.is_main_process:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            _write_json(
                status_path,
                {
                    "status": "running",
                    "step": int(step),
                    "started_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                },
            )
            started = time.monotonic()
            try:
                from .evaluation import run_blocking_meanaudio_957

                result = run_blocking_meanaudio_957(
                    output_dir=self.output_dir,
                    checkpoint=self._checkpoint_dir(step),
                    step=step,
                    config=self.config.audiocaps_eval_config,
                )
                status = {
                    "status": "completed",
                    "step": int(step),
                    "duration_seconds": time.monotonic() - started,
                    "result": result,
                }
            except Exception as exc:
                status = {
                    "status": "failed",
                    "step": int(step),
                    "duration_seconds": time.monotonic() - started,
                    "error": repr(exc),
                }
            status["finished_at"] = (
                datetime.now().astimezone().isoformat(timespec="seconds")
            )
            _write_json(status_path, status)
        else:
            # The main rank's subprocess owns ``timeout_seconds``.  Give it a
            # short grace period to catch that timeout and atomically publish
            # the terminal failed status; otherwise a non-main rank can time
            # out first and leave the distributed job in inconsistent paths.
            deadline = time.monotonic() + float(
                self.config.audiocaps_eval_config.get(
                    "timeout_seconds", 21_600
                )
            ) + float(
                self.config.audiocaps_eval_config.get(
                    "status_wait_grace_seconds", 300
                )
            )
            status: dict[str, Any] = {}
            while time.monotonic() < deadline:
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    status = {}
                if status.get("status") in {"completed", "failed"}:
                    break
                time.sleep(2.0)
            else:
                raise TimeoutError(f"timed out waiting for {status_path}")
        self.accelerator.wait_for_everyone()
        # Every rank reads the same terminal status and takes the same action.
        terminal = json.loads(status_path.read_text(encoding="utf-8"))
        if terminal.get("status") != "completed":
            message = (
                f"blocking AudioCaps-957 evaluation failed at step {step}: "
                f"{terminal.get('error')}"
            )
            failure_policy = str(
                self.config.audiocaps_eval_config.get(
                    "failure_policy", "raise"
                )
            ).strip().lower()
            if failure_policy == "raise":
                raise RuntimeError(message)
            if failure_policy != "continue":
                raise ValueError(
                    "audiocaps_eval.failure_policy must be 'raise' or "
                    f"'continue', got {failure_policy!r}"
                )
            if self.accelerator.is_main_process:
                print(f"[TTA] WARNING: {message}; continuing training")
        if self.discriminator is not None:
            self.discriminator.train()
        self.model.train()


__all__ = ["PMFTrainer"]
