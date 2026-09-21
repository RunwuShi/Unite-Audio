from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


FAD_CACHE_VERSION = 1


def embedding_statistics(embeddings: Tensor) -> tuple[Tensor, Tensor, int]:
    """Return NumPy-compatible (unbiased) Gaussian statistics."""

    values = torch.as_tensor(embeddings, dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("embeddings must have shape [N,D] with N >= 2")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    return mean, covariance, int(values.shape[0])


def frechet_distance(
    mean_a: Tensor,
    covariance_a: Tensor,
    mean_b: Tensor,
    covariance_b: Tensor,
) -> Tensor:
    """Stable symmetric form of the Gaussian Frechet distance."""

    mean_a = torch.as_tensor(mean_a, dtype=torch.float64)
    covariance_a = torch.as_tensor(covariance_a, dtype=torch.float64)
    mean_b = torch.as_tensor(mean_b, dtype=torch.float64, device=mean_a.device)
    covariance_b = torch.as_tensor(
        covariance_b, dtype=torch.float64, device=mean_a.device
    )
    covariance_a = (covariance_a + covariance_a.T) * 0.5
    covariance_b = (covariance_b + covariance_b.T) * 0.5
    eigenvalues, eigenvectors = _stable_eigh(covariance_a)
    square_root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    middle = square_root @ covariance_b @ square_root
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues, _ = _stable_eigh(middle)
    trace_square_root = middle_eigenvalues.clamp_min(0).sqrt().sum()
    difference = mean_a - mean_b
    return (
        difference.square().sum()
        + torch.trace(covariance_a)
        + torch.trace(covariance_b)
        - 2.0 * trace_square_root
    )


def _stable_eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    symmetric = (matrix + matrix.T) * 0.5
    scale = max(1.0, float(torch.diagonal(symmetric).abs().mean().item()))
    identity = torch.eye(
        symmetric.shape[0], device=symmetric.device, dtype=symmetric.dtype
    )
    last_error: RuntimeError | None = None
    for relative_jitter in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
        try:
            values, vectors = torch.linalg.eigh(
                symmetric + identity * (relative_jitter * scale)
            )
            if bool(torch.isfinite(values).all().item()):
                return values, vectors
        except RuntimeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise FloatingPointError("non-finite covariance eigendecomposition")


def statistics_from_sums(
    count: int, feature_sum: Tensor, feature_outer: Tensor
) -> tuple[Tensor, Tensor]:
    if int(count) < 2:
        raise ValueError("at least two embeddings are required")
    mean = feature_sum / int(count)
    covariance = (feature_outer - int(count) * torch.outer(mean, mean)) / (
        int(count) - 1
    )
    return mean, (covariance + covariance.T) * 0.5


def fad_from_sums(
    count: int,
    feature_sum: Tensor,
    feature_outer: Tensor,
    reference_mean: Tensor,
    reference_covariance: Tensor,
) -> Tensor:
    mean, covariance = statistics_from_sums(count, feature_sum, feature_outer)
    return frechet_distance(mean, covariance, reference_mean, reference_covariance)


def marginal_fad_improvement(
    count: int,
    feature_sum: Tensor,
    feature_outer: Tensor,
    candidate: Tensor,
    reference_mean: Tensor,
    reference_covariance: Tensor,
) -> Tensor:
    base = fad_from_sums(
        count, feature_sum, feature_outer, reference_mean, reference_covariance
    )
    values = torch.as_tensor(candidate, dtype=torch.float64, device=feature_sum.device)
    updated = fad_from_sums(
        count + int(values.shape[0]),
        feature_sum + values.sum(dim=0),
        feature_outer + values.T @ values,
        reference_mean,
        reference_covariance,
    )
    return base - updated


class VGGishEmbeddingExtractor:
    """Evaluator-identical VGGish: 16 kHz, 128-D, no PCA, no final ReLU."""

    def __init__(
        self,
        hub_dir: str | Path,
        device: torch.device,
        *,
        batch_size: int = 128,
    ) -> None:
        hub = Path(hub_dir).expanduser().resolve()
        repository = hub / "harritaylor_torchvggish_master"
        checkpoint = hub / "checkpoints" / "vggish-10086976.pth"
        if not repository.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"offline VGGish assets are incomplete: repo={repository}, weights={checkpoint}"
            )
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        # torchvggish imports resampy unconditionally although this protocol is
        # already fixed at 16 kHz.  Keep the environment self-contained and
        # provide a librosa-compatible fallback for diagnostic non-16k inputs.
        try:
            import resampy  # noqa: F401
        except ModuleNotFoundError:
            fallback = types.ModuleType("resampy")
            fallback.__spec__ = ModuleSpec("resampy", loader=None)

            def _resample(data: Any, source_rate: int, target_rate: int) -> Any:
                import librosa

                return librosa.resample(
                    data, orig_sr=source_rate, target_sr=target_rate
                )

            fallback.resample = _resample  # type: ignore[attr-defined]
            sys.modules["resampy"] = fallback
        from torchvggish import vggish_input
        from torchvggish.vggish import VGGish

        model = VGGish(
            urls={},
            device=device,
            pretrained=False,
            preprocess=False,
            postprocess=False,
        )
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.embeddings = nn.Sequential(*list(model.embeddings.children())[:-1])
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = int(batch_size)
        self.waveform_to_examples = vggish_input.waveform_to_examples
        from torchvggish import mel_features

        self.window = torch.hann_window(
            400, periodic=True, device=device, dtype=torch.float32
        )
        self.mel_matrix = torch.as_tensor(
            mel_features.spectrogram_to_mel_matrix(
                num_mel_bins=64,
                num_spectrogram_bins=257,
                audio_sample_rate=16_000,
                lower_edge_hertz=125.0,
                upper_edge_hertz=7_500.0,
            ),
            device=device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def __call__(self, waveform: Tensor) -> Tensor:
        values = waveform
        if values.ndim == 3:
            values = values.mean(dim=1)
        if values.ndim != 2:
            raise ValueError("waveform must have shape [B,L] or [B,C,L]")
        # WaveDataset in the official evaluator subtracts each file's DC mean.
        values = values.float()
        values = values - values.mean(dim=1, keepdim=True)
        # Vectorized torch spelling of torchvggish.mel_features: no centering,
        # 25 ms periodic-Hann windows, 10 ms hop, 512-point magnitude FFT,
        # the original HTK mel matrix, log offset 0.01, and 96-frame patches.
        frames = values.unfold(1, 400, 160)
        magnitude = torch.fft.rfft(frames * self.window, n=512, dim=-1).abs()
        log_mel = torch.log(magnitude @ self.mel_matrix + 0.01)
        patches = log_mel.unfold(1, 96, 96).permute(0, 1, 3, 2).contiguous()
        patch_count = int(patches.shape[1])
        packed = patches.reshape(-1, 1, 96, 64)
        output: list[Tensor] = []
        for start in range(0, packed.shape[0], self.batch_size):
            output.append(self.model(packed[start : start + self.batch_size]).float())
        embeddings = torch.cat(output, dim=0)
        if not bool(torch.isfinite(embeddings).all().item()):
            raise FloatingPointError("VGGish produced non-finite embeddings")
        return embeddings.reshape(values.shape[0], patch_count, 128)


class FADConstraintReward:
    """Queue-based marginal FAD reward with an adaptive Lagrange multiplier."""

    def __init__(
        self,
        cache_path: str | Path,
        hub_dir: str | Path,
        device: torch.device,
        *,
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
                f"missing VGGish FAD cache: {path}; run prepare-fad-cache first"
            )
        try:
            cache = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            cache = torch.load(path, map_location="cpu")
        if int(cache.get("version", -1)) != FAD_CACHE_VERSION:
            raise ValueError(f"unsupported VGGish FAD cache version in {path}")
        self.extractor = VGGishEmbeddingExtractor(
            hub_dir, device, batch_size=embedding_batch_size
        )
        self.device = device
        self.queue_capacity = int(queue_capacity)
        self.dual_lr = float(dual_lr)
        self.lambda_max = float(lambda_max)
        self.exact_every = int(exact_every)
        self.lagrange = float(lambda_init)
        self.target = float(cache["baseline_fad"]) * float(target_scale)
        self.reference_count = int(cache["reference_count"])
        self.reference_sum = torch.as_tensor(
            cache["reference_sum"], dtype=torch.float64, device=device
        )
        self.reference_outer = torch.as_tensor(
            cache["reference_outer"], dtype=torch.float64, device=device
        )
        self.reference_mean, self.reference_covariance = statistics_from_sums(
            self.reference_count, self.reference_sum, self.reference_outer
        )
        reference_values, reference_vectors = _stable_eigh(self.reference_covariance)
        self.reference_sqrt = (
            reference_vectors * reference_values.clamp_min(0).sqrt()
        ) @ reference_vectors.T
        baseline = torch.as_tensor(cache["baseline_audio_embeddings"]).float()
        if baseline.ndim != 3 or baseline.shape[-1] != 128:
            raise ValueError("baseline_audio_embeddings must have shape [N,P,128]")
        if baseline.shape[0] < self.queue_capacity:
            raise ValueError(
                f"FAD cache has {baseline.shape[0]} baseline audios, "
                f"but queue_capacity={self.queue_capacity}"
            )
        self.queue: deque[Tensor] = deque(maxlen=self.queue_capacity)
        for row in baseline[-self.queue_capacity :]:
            self.queue.append(row.contiguous().cpu())
        self._rebuild_sums()
        self.proxy = self._queue_fad()
        self.steps = 0

    def _rebuild_sums(self) -> None:
        flattened = torch.cat(list(self.queue), dim=0).to(
            self.device, dtype=torch.float64
        )
        self.queue_count = int(flattened.shape[0])
        self.queue_sum = flattened.sum(dim=0)
        self.queue_outer = flattened.T @ flattened

    def _fad_with(self, extra: Tensor | None = None) -> float:
        count = self.queue_count
        feature_sum = self.queue_sum
        feature_outer = self.queue_outer
        if extra is not None:
            values = extra.to(self.device, dtype=torch.float64)
            count += int(values.shape[0])
            feature_sum = feature_sum + values.sum(dim=0)
            feature_outer = feature_outer + values.T @ values
        value = fad_from_sums(
            count,
            feature_sum,
            feature_outer,
            self.reference_mean,
            self.reference_covariance,
        )
        return float(value.clamp_min(0).item())

    def _queue_fad(self) -> float:
        return self._fad_with(None)

    def _batched_marginal(self, candidates: Tensor, base_fad: float) -> Tensor:
        values = candidates.to(self.device, dtype=torch.float64)
        batch, patches, dimensions = values.shape
        count = self.queue_count + patches
        sums = self.queue_sum[None, :] + values.sum(dim=1)
        outers = self.queue_outer[None, :, :] + torch.einsum(
            "bpd,bpe->bde", values, values
        )
        means = sums / count
        covariances = (outers - count * torch.einsum("bd,be->bde", means, means)) / (
            count - 1
        )
        covariances = (covariances + covariances.transpose(1, 2)) * 0.5
        middle = self.reference_sqrt[None] @ covariances @ self.reference_sqrt[None]
        middle = (middle + middle.transpose(1, 2)) * 0.5
        scale = torch.diagonal(middle, dim1=1, dim2=2).abs().mean(dim=1)
        identity = torch.eye(dimensions, device=self.device, dtype=torch.float64)
        middle = middle + scale[:, None, None].clamp_min(1.0) * 1.0e-10 * identity
        eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0)
        differences = means - self.reference_mean[None]
        distances = (
            differences.square().sum(dim=1)
            + torch.diagonal(covariances, dim1=1, dim2=2).sum(dim=1)
            + torch.trace(self.reference_covariance)
            - 2.0 * eigenvalues.sqrt().sum(dim=1)
        )
        return torch.as_tensor(base_fad, device=self.device) - distances.float()

    def _gather_audio_embeddings(self, local: Tensor) -> list[Tensor]:
        local = local.detach().float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [
                torch.empty_like(local)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered, local.contiguous())
            packed = torch.cat(gathered, dim=0)
        else:
            packed = local
        return [row.cpu() for row in packed]

    @torch.inference_mode()
    def __call__(self, waveform: Tensor) -> dict[str, Tensor]:
        embeddings = self.extractor(waveform)
        base_fad = self._queue_fad()
        # Positive means this sample moves the queue closer to real AudioCaps.
        marginal = self._batched_marginal(embeddings, base_fad).to(waveform.device)

        for candidate in self._gather_audio_embeddings(embeddings):
            if len(self.queue) == self.queue_capacity:
                removed = self.queue.popleft().to(self.device, dtype=torch.float64)
                self.queue_count -= int(removed.shape[0])
                self.queue_sum -= removed.sum(dim=0)
                self.queue_outer -= removed.T @ removed
            self.queue.append(candidate)
            added = candidate.to(self.device, dtype=torch.float64)
            self.queue_count += int(added.shape[0])
            self.queue_sum += added.sum(dim=0)
            self.queue_outer += added.T @ added

        if (self.steps + 1) % self.exact_every == 0:
            # Periodically remove any roundoff accumulated by FIFO subtraction.
            self._rebuild_sums()
        self.proxy = self._queue_fad()
        relative_violation = (self.proxy - self.target) / max(self.target, 1.0e-8)
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            self.lagrange = min(
                self.lambda_max,
                max(0.0, self.lagrange + self.dual_lr * relative_violation),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            synchronized = torch.tensor(
                [self.proxy, self.lagrange], device=self.device, dtype=torch.float64
            )
            torch.distributed.broadcast(synchronized, src=0)
            self.proxy, self.lagrange = map(float, synchronized.tolist())
            relative_violation = (self.proxy - self.target) / max(self.target, 1.0e-8)
        self.steps += 1
        result = self.status(waveform.shape[0], waveform.device)
        result["marginal"] = marginal
        return result

    def status(self, batch: int, device: torch.device) -> dict[str, Tensor]:
        relative_violation = (self.proxy - self.target) / max(self.target, 1.0e-8)
        return {
            "marginal": torch.zeros(int(batch), device=device),
            "proxy": torch.full((int(batch),), self.proxy, device=device),
            "target": torch.full((int(batch),), self.target, device=device),
            "lambda": torch.full((int(batch),), self.lagrange, device=device),
            "violation": torch.full((int(batch),), relative_violation, device=device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": FAD_CACHE_VERSION,
            "queue_embeddings": torch.stack(list(self.queue), dim=0),
            "lagrange": self.lagrange,
            "proxy": self.proxy,
            "steps": self.steps,
            "target": self.target,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != FAD_CACHE_VERSION:
            raise ValueError("unsupported FAD reward checkpoint version")
        if abs(float(state["target"]) - self.target) > 1.0e-6:
            raise ValueError("FAD target differs between checkpoint and cache")
        values = torch.as_tensor(state["queue_embeddings"]).float()
        if values.shape[0] != self.queue_capacity:
            raise ValueError("checkpoint FAD queue capacity does not match config")
        self.queue.clear()
        self.queue.extend(row.contiguous().cpu() for row in values)
        self.lagrange = float(state["lagrange"])
        self.proxy = float(state["proxy"])
        self.steps = int(state["steps"])
        self._rebuild_sums()


__all__ = [
    "FAD_CACHE_VERSION",
    "FADConstraintReward",
    "VGGishEmbeddingExtractor",
    "embedding_statistics",
    "fad_from_sums",
    "frechet_distance",
    "marginal_fad_improvement",
    "statistics_from_sums",
]
