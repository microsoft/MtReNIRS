# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public API for the MtSi1005 montage alignment toolkit."""

from .alignment import align, align_optodes, load_reference
from .io import from_mne_raw, load_snirf, load_xdf
from .models import AlignmentResult, MontageInput


class MtSi1005:
    """Namespace for the MtSi1005 montage alignment operations."""

    align = staticmethod(align)
    align_optodes = staticmethod(align_optodes)
    load_reference = staticmethod(load_reference)


__all__ = [
    "AlignmentResult",
    "MtSi1005",
    "MontageInput",
    "align",
    "align_optodes",
    "from_mne_raw",
    "load_reference",
    "load_snirf",
    "load_xdf",
]

__version__ = "0.1.0"
