# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pytest

from mtrenirs.models.segmentation import extract_windows, minimum_overlap_starts, window_starts


@pytest.mark.parametrize(
    ("length", "expected"),
    [(100, [0]), (130, [0, 30]), (200, [0, 100]), (250, [0, 75, 150])],
)
def test_minimum_overlap_starts(length: int, expected: list[int]) -> None:
    assert minimum_overlap_starts(length).tolist() == expected


def test_window_starts_defaults_to_non_overlapping_drop() -> None:
    assert window_starts(630).tolist() == [0, 100, 200, 300, 400, 500]
    assert window_starts(630, remainder="pad").tolist() == [0, 100, 200, 300, 400, 500, 600]
    assert window_starts(630, remainder="overlap").tolist() == [0, 88, 177, 265, 353, 442, 530]


def test_extract_windows_remainder_policy() -> None:
    signal = np.arange(2 * 130).reshape(2, 130)
    windows, starts = extract_windows(signal, onset=0, length=130)
    assert starts.tolist() == [0]
    np.testing.assert_array_equal(windows[0], signal[:, :100])

    windows, starts = extract_windows(signal, onset=0, length=130, remainder="overlap")
    assert starts.tolist() == [0, 30]
    np.testing.assert_array_equal(windows[0], signal[:, :100])
    np.testing.assert_array_equal(windows[1], signal[:, 30:130])

    windows, starts = extract_windows(signal, onset=0, length=130, remainder="pad")
    assert starts.tolist() == [0, 100]
    np.testing.assert_array_equal(windows[0], signal[:, :100])
    np.testing.assert_array_equal(windows[1, :, :30], signal[:, 100:130])
    np.testing.assert_array_equal(windows[1, :, 30:], 0)
