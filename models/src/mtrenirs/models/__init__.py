# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public API for the MtReNIRS foundation model."""

from .model import EmbeddingResult, MtReNIRS
from .recording import Recording
from .segmentation import minimum_overlap_starts, window_starts

__all__ = [
	"EmbeddingResult",
	"MtReNIRS",
	"Recording",
	"minimum_overlap_starts",
	"window_starts",
]
__version__ = "0.1.0"
