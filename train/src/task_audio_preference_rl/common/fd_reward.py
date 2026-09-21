from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


FD_CACHE_VERSION = 1


def diagonal_statistics_from_sums(
    count: int, feature_sum: Tensor, feature_square_sum: Tensor
) -> tuple[Tensor, Tensor]:
    if int(count) < 2:
        raise ValueError("at least two embeddings are required")
    mean = feature_sum / int(count)
    variance = (feature_square_sum - int(count) * mean.square()) / (int(count) - 1)
    return mean, variance.clamp_min(0)


def diagonal_frechet_distance(
    mean_a: Tensor,
    variance_a: Tensor,
    mean_b: Tensor,
    variance_b: Tensor,
) -> Tensor:
    mean_a = torch.as_tensor(mean_a, dtype=torch.float64)
    variance_a = torch.as_tensor(variance_a, dtype=torch.float64, device=mean_a.device)
    mean_b = torch.as_tensor(mean_b, dtype=torch.float64, device=mean_a.device)
    variance_b = torch.as_tensor(
        variance_b, dtype=torch.float64, device=mean_a.device
    )
    covariance_term = (
        variance_a.clamp_min(0)
        + variance_b.clamp_min(0)
        - 2.0 * (variance_a.clamp_min(0) * variance_b.clamp_min(0)).sqrt()
    )
    return (mean_a - mean_b).square().sum() + covariance_term.sum()


class PANNsEmbeddingExtractor:
    """Evaluator-identical AudioLDM PANNs CNN14 2048-D clip embeddings."""

    def __init__(
        self,
        repository: str | Path,
        checkpoint_dir: str | Path,
        device: torch.device,
        *,
        batch_size: int = 16,
    ) -> None:
        repo = Path(repository).expanduser().resolve()
        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        if not (repo / "audioldm_eval/feature_extractors/panns").is_dir():
            raise FileNotFoundError(f"missing AudioLDM evaluation repository: {repo}")
        expected = checkpoint / "Cnn14_16k_mAP=0.438.pth"
        if not expected.is_file():
            raise FileNotFoundError(f"missing PANNs checkpoint: {expected}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        os.environ["AUDIOLDM_EVAL_CHECKPOINT_DIR"] = str(checkpoint)
        from audioldm_eval.feature_extractors.panns import Cnn14

        self.model = Cnn14(
            features_list=["2048"],
            sample_rate=16_000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8_000,
            classes_num=527,
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def __call__(self, waveform: Tensor) -> Tensor:
        values = waveform
        if values.ndim == 3:
            values = values.mean(dim=1)
        if values.ndim != 2:
            raise ValueError("waveform must have shape [B,L] or [B,C,L]")
        values = values.float()
        values = values - values.mean(dim=1, keepdim=True)
        if values.shape[1] < 160_000:
            values = F.pad(values, (0, 160_000 - values.shape[1]))
        else:
            values = values[:, :160_000]
        outputs: list[Tensor] = []
        for start in range(0, values.shape[0], self.batch_size):
            result = self.model(values[start : start + self.batch_size].to(self.device))
            outputs.append(torch.as_tensor(result["2048"]).float())
        embeddings = torch.cat(outputs, dim=0)
        if embeddings.shape != (values.shape[0], 2048):
            raise RuntimeError(f"unexpected PANNs embedding shape: {embeddings.shape}")
        if not bool(torch.isfinite(embeddings).all().item()):
            raise FloatingPointError("PANNs produced non-finite embeddings")
        return embeddings


class ProjectedFDConstraintReward:
    """Queue-based marginal PANNs FD in a fixed PCA-diagonal feature space."""

    def __init__(
        self,
        cache_path: str | Path,
        panns_repo: str | Path,
        panns_checkpoint_dir: str | Path,
        device: torch.device,
        *,
        projection_dim: int,
        queue_capacity: int,
        target_scale: float,
        dual_lr: float,
        lambda_init: float,
        lambda_max: float,
        exact_every: int,
        embedding_batch_size: int,
    ) -> None:
        path = Path(cache_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"missing PANNs FD cache: {path}; run prepare-constraint-caches first"
            )
        try:
            cache = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            cache = torch.load(path, map_location="cpu")
        if int(cache.get("version", -1)) != FD_CACHE_VERSION:
            raise ValueError(f"unsupported PANNs FD cache version in {path}")
        components = torch.as_tensor(cache["components"]).float()
        if components.shape != (int(projection_dim), 2048):
            raise ValueError(
                f"FD cache projection is {tuple(components.shape)}, expected "
                f"({projection_dim}, 2048)"
            )
        self.extractor = PANNsEmbeddingExtractor(
            panns_repo,
            panns_checkpoint_dir,
            device,
            batch_size=embedding_batch_size,
        )
        self.device = device
        self.center = torch.as_tensor(cache["center"]).float().to(device)
        self.components = components.to(device)
        self.reference_mean = torch.as_tensor(
            cache["reference_mean"], dtype=torch.float64, device=device
        )
        self.reference_variance = torch.as_tensor(
            cache["reference_variance"], dtype=torch.float64, device=device
        )
        self.queue_capacity = int(queue_capacity)
        self.dual_lr = float(dual_lr)
        self.lambda_max = float(lambda_max)
        self.exact_every = int(exact_every)
        self.lagrange = float(lambda_init)
        self.target = float(cache["baseline_fd"]) * float(target_scale)
        baseline = torch.as_tensor(cache["baseline_embeddings"]).float()
        if baseline.ndim != 2 or baseline.shape[1] != int(projection_dim):
            raise ValueError("baseline_embeddings must have shape [N,projection_dim]")
        if baseline.shape[0] < self.queue_capacity:
            raise ValueError(
                f"FD cache has {baseline.shape[0]} baseline audios, "
                f"but queue_capacity={self.queue_capacity}"
            )
        self.queue: deque[Tensor] = deque(maxlen=self.queue_capacity)
        self.queue.extend(row.contiguous().cpu() for row in baseline[-self.queue_capacity :])
        self._rebuild_sums()
        self.proxy = self._queue_fd()
        self.steps = 0

    def _project(self, embeddings: Tensor) -> Tensor:
        return (embeddings.float() - self.center) @ self.components.T

    def _rebuild_sums(self) -> None:
        values = torch.stack(list(self.queue), dim=0).to(
            self.device, dtype=torch.float64
        )
        self.queue_count = int(values.shape[0])
        self.queue_sum = values.sum(dim=0)
        self.queue_square_sum = values.square().sum(dim=0)

    def _fd_from_sums(self, count: int, feature_sum: Tensor, square_sum: Tensor) -> Tensor:
        mean, variance = diagonal_statistics_from_sums(count, feature_sum, square_sum)
        return diagonal_frechet_distance(
            mean, variance, self.reference_mean, self.reference_variance
        )

    def _queue_fd(self) -> float:
        value = self._fd_from_sums(
            self.queue_count, self.queue_sum, self.queue_square_sum
        )
        return float(value.clamp_min(0).item())

    def _batched_marginal(self, candidates: Tensor, base_fd: float) -> Tensor:
        values = candidates.to(self.device, dtype=torch.float64)
        count = self.queue_count + 1
        sums = self.queue_sum[None] + values
        square_sums = self.queue_square_sum[None] + values.square()
        means = sums / count
        variances = (square_sums - count * means.square()) / (count - 1)
        distances = (means - self.reference_mean[None]).square().sum(dim=1)
        distances = distances + (
            variances.clamp_min(0)
            + self.reference_variance[None].clamp_min(0)
            - 2.0
            * (
                variances.clamp_min(0)
                * self.reference_variance[None].clamp_min(0)
            ).sqrt()
        ).sum(dim=1)
        return torch.as_tensor(base_fd, device=self.device) - distances.float()

    def _gather_embeddings(self, local: Tensor) -> list[Tensor]:
        local = local.detach().float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [
                torch.empty_like(local) for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered, local.contiguous())
            packed = torch.cat(gathered, dim=0)
        else:
            packed = local
        return [row.cpu() for row in packed]

    def _status(self, batch: int, device: torch.device) -> dict[str, Tensor]:
        violation = (self.proxy - self.target) / max(self.target, 1.0e-8)
        return {
            "marginal": torch.zeros(batch, device=device),
            "proxy": torch.full((batch,), self.proxy, device=device),
            "target": torch.full((batch,), self.target, device=device),
            "lambda": torch.full((batch,), self.lagrange, device=device),
            "violation": torch.full((batch,), violation, device=device),
        }

    @torch.inference_mode()
    def __call__(self, waveform: Tensor) -> dict[str, Tensor]:
        embeddings = self._project(self.extractor(waveform))
        base_fd = self._queue_fd()
        marginal = self._batched_marginal(embeddings, base_fd).to(waveform.device)
        for candidate in self._gather_embeddings(embeddings):
            if len(self.queue) == self.queue_capacity:
                removed = self.queue.popleft().to(self.device, dtype=torch.float64)
                self.queue_count -= 1
                self.queue_sum -= removed
                self.queue_square_sum -= removed.square()
            self.queue.append(candidate)
            added = candidate.to(self.device, dtype=torch.float64)
            self.queue_count += 1
            self.queue_sum += added
            self.queue_square_sum += added.square()
        if (self.steps + 1) % self.exact_every == 0:
            self._rebuild_sums()
        self.proxy = self._queue_fd()
        violation = (self.proxy - self.target) / max(self.target, 1.0e-8)
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            self.lagrange = min(
                self.lambda_max,
                max(0.0, self.lagrange + self.dual_lr * violation),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            synchronized = torch.tensor(
                [self.proxy, self.lagrange], device=self.device, dtype=torch.float64
            )
            torch.distributed.broadcast(synchronized, src=0)
            self.proxy, self.lagrange = map(float, synchronized.tolist())
            violation = (self.proxy - self.target) / max(self.target, 1.0e-8)
        self.steps += 1
        result = self._status(waveform.shape[0], waveform.device)
        result["marginal"] = marginal
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": FD_CACHE_VERSION,
            "queue_embeddings": torch.stack(list(self.queue), dim=0),
            "lagrange": self.lagrange,
            "proxy": self.proxy,
            "steps": self.steps,
            "target": self.target,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != FD_CACHE_VERSION:
            raise ValueError("unsupported FD reward checkpoint version")
        if abs(float(state["target"]) - self.target) > 1.0e-6:
            raise ValueError("FD target differs between checkpoint and cache")
        values = torch.as_tensor(state["queue_embeddings"]).float()
        if values.shape[0] != self.queue_capacity:
            raise ValueError("checkpoint FD queue capacity does not match config")
        self.queue.clear()
        self.queue.extend(row.contiguous().cpu() for row in values)
        self.lagrange = float(state["lagrange"])
        self.proxy = float(state["proxy"])
        self.steps = int(state["steps"])
        self._rebuild_sums()


__all__ = [
    "FD_CACHE_VERSION",
    "PANNsEmbeddingExtractor",
    "ProjectedFDConstraintReward",
    "diagonal_frechet_distance",
    "diagonal_statistics_from_sums",
]
