from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .network import MtReNIRSEncoder
from .recording import Recording
from .segmentation import extract_windows, window_starts


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: np.ndarray
    window_onsets_seconds: np.ndarray
    window_duration_seconds: float
    pooling: str


class MtReNIRS:
    def __init__(
        self,
        encoder: MtReNIRSEncoder,
        config: dict,
        device: str,
        *,
        ch_types: tuple[str, ...] | None = None,
        ch_pos_1005: np.ndarray | None = None,
        ch_pos_1005_valid: np.ndarray | None = None,
    ):
        self.encoder = encoder.eval().to(device)
        self.config = config
        self.device = torch.device(device)
        self.sample_rate = float(config["data"]["sfreq"])
        self.window_samples = int(config["data"]["t_win"])
        self.window_seconds = float(config["data"]["window_sec"])
        self.embedding_dim = int(config["model"]["d_model"])
        self.ch_types = ch_types
        self.ch_pos_1005 = ch_pos_1005
        self.ch_pos_1005_valid = ch_pos_1005_valid

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "mtrenirs-10s",
        device: str | None = None,
        *,
        ch_types: list[str] | tuple[str, ...] | None = None,
        ch_pos_1005: np.ndarray | torch.Tensor | None = None,
        ch_pos_1005_valid: np.ndarray | torch.Tensor | None = None,
    ):
        path = cls._resolve_checkpoint(checkpoint)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        config = loaded["config"]
        encoder = MtReNIRSEncoder(config)
        state = {
            key: value
            for key, value in loaded["model"].items()
            if key.startswith("embed.") or key.startswith("encoder.")
        }
        encoder.load_state_dict(state, strict=True)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        positions = None if ch_pos_1005 is None else np.asarray(
            torch.as_tensor(ch_pos_1005).cpu(), dtype=np.float32
        )
        valid = None if ch_pos_1005_valid is None else np.asarray(
            torch.as_tensor(ch_pos_1005_valid).cpu(), dtype=bool
        )
        types = None if ch_types is None else tuple(ch_types)
        if (types is None) != (positions is None):
            raise ValueError("provide both ch_types and ch_pos_1005")
        if positions is not None:
            if positions.shape != (len(types), 3):
                raise ValueError("ch_pos_1005 must have shape (channels, 3)")
            if valid is None:
                valid = np.isfinite(positions).all(axis=1)
            if valid.shape != (len(types),):
                raise ValueError("ch_pos_1005_valid must have shape (channels,)")
        return cls(
            encoder,
            config,
            selected_device,
            ch_types=types,
            ch_pos_1005=positions,
            ch_pos_1005_valid=valid,
        )

    @staticmethod
    def _resolve_checkpoint(checkpoint: str | Path) -> Path:
        path = Path(checkpoint).expanduser()
        if path.is_file():
            return path
        if str(checkpoint) == "mtrenirs-10s":
            candidates = (
                Path(__file__).resolve().parent / "weights" / "mtrenirs_10s.pt",
                Path(__file__).resolve().parents[3] / "weights" / "mtrenirs_10s.pt",
            )
            for bundled in candidates:
                if bundled.is_file():
                    return bundled
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    def _coordinates(self, recording: Recording) -> np.ndarray:
        return self._coordinate_array(
            recording.ch_types,
            recording.ch_pos_1005,
            recording.ch_pos_1005_valid,
        )

    def _coordinate_array(
        self,
        ch_types: tuple[str, ...],
        ch_pos_1005: np.ndarray,
        ch_pos_1005_valid: np.ndarray,
    ) -> np.ndarray:
        positions = ch_pos_1005.copy()
        valid = ch_pos_1005_valid & np.isfinite(positions).all(axis=1)
        center = np.asarray(self.config["data"]["pos_center"], dtype=np.float32)
        scale = float(self.config["data"]["pos_scale"])
        positions[valid] = (positions[valid] - center) / scale
        positions[~valid] = 0.0
        chromophore = np.asarray(
            [self.config["data"]["chromophore_map"].get(kind.lower(), -1) for kind in ch_types],
            dtype=np.float32,
        )
        if (chromophore < 0).any():
            unknown = sorted({kind for kind, value in zip(ch_types, chromophore) if value < 0})
            raise ValueError(f"unsupported channel types: {unknown}")
        return np.column_stack([positions, chromophore])

    @torch.no_grad()
    def embed(
        self,
        signal: torch.Tensor,
        *,
        pool: str = "none",
        remainder: str = "drop",
    ) -> torch.Tensor:
        if self.ch_types is None or self.ch_pos_1005 is None or self.ch_pos_1005_valid is None:
            raise ValueError("bind ch_types and ch_pos_1005 in from_pretrained()")
        signal = torch.as_tensor(signal, dtype=torch.float32)
        if signal.ndim == 2:
            signal = signal.unsqueeze(0)
        if signal.ndim != 3:
            raise ValueError("signal must have shape (batch, channels, samples) or (channels, samples)")
        batch, channels, samples = signal.shape
        if channels != len(self.ch_types):
            raise ValueError("signal channels do not match the bound montage")
        starts = window_starts(samples, self.window_samples, remainder)
        split_windows = []
        for start in starts:
            window = signal[:, :, start : start + self.window_samples]
            if window.shape[-1] < self.window_samples:
                padded = signal.new_zeros(batch, channels, self.window_samples)
                padded[:, :, : window.shape[-1]] = window
                window = padded
            split_windows.append(window)
        windows = torch.stack(split_windows, dim=1).to(self.device)
        window_count = windows.shape[1]
        tokens = windows.reshape(batch, window_count * channels, self.window_samples)
        coordinate = self._coordinate_array(
            self.ch_types,
            self.ch_pos_1005,
            self.ch_pos_1005_valid,
        )
        coordinates = torch.from_numpy(np.tile(coordinate, (batch, window_count, 1))).to(self.device)
        indices = torch.arange(window_count, device=self.device).repeat_interleave(channels)
        window_indices = indices.repeat(batch, 1)
        channel_valid = torch.from_numpy(np.tile(self.ch_pos_1005_valid, window_count)).to(self.device)
        valid = channel_valid.repeat(batch, 1)
        encoded = self.encoder(tokens, coordinates, window_indices, valid)
        encoded = encoded.reshape(batch, window_count, channels, self.embedding_dim)
        if pool == "none":
            return encoded
        if pool == "flatten":
            return encoded.reshape(batch, window_count * channels * self.embedding_dim)
        if pool == "temporal_mean":
            return encoded.mean(dim=1)
        raise ValueError("pool must be none, flatten, or temporal_mean")

    @torch.no_grad()
    def embed_windows(
        self,
        windows: np.ndarray,
        recording: Recording,
        *,
        pool: str = "none",
    ) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 3:
            windows = windows[None]
        if windows.ndim != 4:
            raise ValueError("windows must have shape (windows, channels, samples) or batched equivalent")
        batch, window_count, channels, samples = windows.shape
        if channels != recording.signal.shape[0] or samples != self.window_samples:
            raise ValueError("window channel/sample dimensions do not match the model and recording")
        signal = torch.from_numpy(windows.reshape(batch, window_count * channels, samples)).to(self.device)
        coordinate = self._coordinates(recording)
        coordinates = np.tile(coordinate, (batch, window_count, 1))
        coordinates_tensor = torch.from_numpy(coordinates.astype(np.float32)).to(self.device)
        indices = np.repeat(np.arange(window_count), channels)
        window_indices = torch.from_numpy(np.tile(indices, (batch, 1))).long().to(self.device)
        channel_valid = np.tile(recording.ch_pos_1005_valid, window_count)
        valid = torch.from_numpy(np.tile(channel_valid, (batch, 1))).bool().to(self.device)
        encoded = self.encoder(signal, coordinates_tensor, window_indices, valid)
        encoded = encoded.reshape(batch, window_count, channels, self.embedding_dim)
        if pool == "none":
            output = encoded
        elif pool == "flatten":
            output = encoded.reshape(batch, -1)
        elif pool == "temporal_mean":
            output = encoded.mean(dim=1)
        else:
            raise ValueError("pool must be none, flatten, or temporal_mean")
        return output.cpu().numpy()

    def embed_segment(
        self,
        recording: Recording,
        *,
        onset: float = 0.0,
        duration: float | None = None,
        pool: str = "none",
        short_policy: str = "error",
        remainder: str = "drop",
    ) -> EmbeddingResult:
        if not np.isclose(recording.sfreq, self.sample_rate):
            raise ValueError(f"recording must be sampled at {self.sample_rate:g} Hz")
        if not recording.normalized:
            raise ValueError("recording must be run-normalized before embedding")
        onset_samples = int(round(onset * recording.sfreq))
        available = recording.signal.shape[1] - onset_samples
        length = available if duration is None else int(round(duration * recording.sfreq))
        windows, starts = extract_windows(
            recording.signal,
            onset=onset_samples,
            length=length,
            window=self.window_samples,
            short_policy=short_policy,
            remainder=remainder,
        )
        embeddings = self.embed_windows(windows, recording, pool=pool)[0]
        return EmbeddingResult(
            embeddings=embeddings,
            window_onsets_seconds=starts / recording.sfreq,
            window_duration_seconds=self.window_seconds,
            pooling=pool,
        )

    def embed_recording(
        self,
        recording: Recording,
        *,
        pool: str = "none",
        remainder: str = "drop",
    ) -> EmbeddingResult:
        return self.embed_segment(recording, pool=pool, remainder=remainder)
