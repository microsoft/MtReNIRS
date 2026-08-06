import numpy as np
import torch

from mtrenirs.models import MtReNIRS
from mtrenirs.models.network import MtReNIRSEncoder
from mtrenirs.models.recording import Recording


def _config() -> dict:
    return {
        "data": {
            "t_win": 100,
            "sfreq": 10.0,
            "window_sec": 10.0,
            "pos_center": [0.0, 0.0, 0.0],
            "pos_scale": 1.0,
            "chromophore_map": {"hbo": 0, "hbr": 1},
        },
        "model": {
            "d_model": 16,
            "fourier_input_dim": 4,
            "fourier_num_bands": 2,
            "spatial_mlp_hidden": 8,
            "rope_base": 500.0,
            "qk_norm": True,
            "encoder": {"depth": 2, "n_heads": 4, "mlp_hidden": 32, "dropout": 0.0},
        },
    }


def test_encoder_output_shape() -> None:
    encoder = MtReNIRSEncoder(_config()).eval()
    signal = torch.randn(2, 6, 100)
    coordinates = torch.randn(2, 6, 4)
    indices = torch.tensor([[0, 0, 0, 1, 1, 1]]).repeat(2, 1)
    valid = torch.ones(2, 6, dtype=torch.bool)
    output = encoder(signal, coordinates, indices, valid)
    assert output.shape == (2, 6, 16)


def test_recording_validates_channel_axes() -> None:
    recording = Recording(
        signal=np.zeros((2, 100)),
        sfreq=10,
        ch_names=("a", "b"),
        ch_types=("hbo", "hbr"),
        ch_pos_1005=np.zeros((2, 3)),
        normalized=True,
    )
    assert recording.ch_pos_1005_valid.tolist() == [True, True]


def test_default_checkpoint_is_bundled() -> None:
    checkpoint = MtReNIRS._resolve_checkpoint("mtrenirs-10s")
    assert checkpoint.name == "mtrenirs_10s.pt"
    assert checkpoint.parent.name == "weights"
    assert checkpoint.stat().st_size > 0

    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(contents) == {"config", "model"}
    assert set(contents["config"]) == {"data", "model"}
    assert set(contents["config"]["data"]) == {
        "sfreq",
        "window_sec",
        "t_win",
        "chromophore_map",
        "pos_center",
        "pos_scale",
    }
    assert set(contents["config"]["model"]) == {
        "d_model",
        "fourier_num_bands",
        "fourier_input_dim",
        "spatial_mlp_hidden",
        "rope_base",
        "qk_norm",
        "encoder",
    }


def test_recording_npz_requires_canonical_signal(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, X=np.zeros((1, 2, 100), dtype=np.float32))
    try:
        Recording.from_npz(path)
    except ValueError as error:
        assert "signal" in str(error)
    else:
        raise AssertionError("legacy X input must not be inferred as a continuous recording")


def test_torch_embed_shapes() -> None:
    config = _config()
    model = MtReNIRS(
        MtReNIRSEncoder(config),
        config,
        "cpu",
        ch_types=("hbo", "hbr"),
        ch_pos_1005=np.zeros((2, 3), dtype=np.float32),
        ch_pos_1005_valid=np.ones(2, dtype=bool),
    )
    signal = torch.randn(3, 2, 130)
    default = model.embed(signal)
    unpooled = model.embed(signal, pool="none", remainder="overlap")
    flattened = model.embed(signal, pool="flatten", remainder="overlap")
    temporal_mean = model.embed(signal, pool="temporal_mean", remainder="overlap")
    padded = model.embed(signal, remainder="pad")
    assert default.shape == (3, 1, 2, 16)
    assert unpooled.shape == (3, 2, 2, 16)
    assert flattened.shape == (3, 64)
    assert temporal_mean.shape == (3, 2, 16)
    assert padded.shape == (3, 2, 2, 16)
    torch.testing.assert_close(flattened, unpooled.reshape(3, 64))
    torch.testing.assert_close(temporal_mean, unpooled.mean(dim=1))

    recording = Recording(
        signal=np.zeros((2, 130), dtype=np.float32),
        sfreq=10,
        ch_names=("a", "b"),
        ch_types=("hbo", "hbr"),
        ch_pos_1005=np.zeros((2, 3), dtype=np.float32),
        normalized=True,
    )
    windows = np.zeros((2, 2, 100), dtype=np.float32)
    assert model.embed_windows(windows, recording, pool="none").shape == (1, 2, 2, 16)
    result = model.embed_recording(recording)
    assert result.embeddings.shape == (1, 2, 16)
    padded_result = model.embed_recording(recording, remainder="pad")
    assert padded_result.embeddings.shape == (2, 2, 16)
    assert padded_result.window_onsets_seconds.tolist() == [0.0, 10.0]
