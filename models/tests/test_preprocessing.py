# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mtrenirs.models.adapters.muse_s_athena import _load_optics, load_muse_s_athena_xdf


def _muse_stream(sample_count: int = 2560, sample_rate: float = 64.0) -> dict:
    timestamps = np.arange(sample_count, dtype=np.float64) / sample_rate
    time_series = np.zeros((sample_count, 16), dtype=np.float64)
    for channel in range(8):
        time_series[:, channel] = 1000.0 + 20.0 * np.sin(
            2.0 * np.pi * (0.05 + channel * 0.005) * timestamps
        )
    return {
        "info": {
            "name": ["Muse-OPTICS (00:00:00:00:00:00)"],
            "type": ["PPG"],
            "channel_count": ["16"],
        },
        "time_series": time_series,
        "time_stamps": timestamps,
    }


def test_load_muse_optics_uses_name_and_effective_sample_rate(monkeypatch):
    stream = _muse_stream()
    monkeypatch.setitem(
        sys.modules,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda *args, **kwargs: ([stream], {})),
    )

    intensity, sample_rate = _load_optics("recording.xdf")

    assert intensity.shape == (8, 2560)
    assert sample_rate == pytest.approx(64.0)


def test_muse_adapter_preprocesses_and_aligns_channels(monkeypatch):
    stream = _muse_stream()
    monkeypatch.setitem(
        sys.modules,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda *args, **kwargs: ([stream], {})),
    )

    recording = load_muse_s_athena_xdf("recording.xdf")

    assert recording.signal.shape == (8, 400)
    assert recording.sfreq == 10.0
    assert recording.normalized
    assert recording.ch_names == (
        "S1_D1 hbo",
        "S1_D1 hbr",
        "S1_D2 hbo",
        "S1_D2 hbr",
        "S1_D3 hbo",
        "S1_D3 hbr",
        "S1_D4 hbo",
        "S1_D4 hbr",
    )
    assert recording.ch_types == ("hbo", "hbr") * 4
    assert recording.ch_pos_1005_valid.all()
    np.testing.assert_array_equal(recording.ch_pos_1005[0::2], recording.ch_pos_1005[1::2])
    assert np.isfinite(recording.signal).all()
