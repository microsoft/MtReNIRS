# pyright: reportMissingImports=false

import numpy as np
import torch

from mtrenirs.alignment import MtSi1005
from mtrenirs.models import MtReNIRS

device = "cuda" if torch.cuda.is_available() else "cpu"
ch_types = ("hbo", "hbr") * 4

# Public Standard 1005 slots for a synthetic example. Real inputs should use
# montage alignment in signal order.
reference = MtSi1005.load_reference("standard_1005")
slot_lookup = dict(zip(reference["labels"], reference["positions"]))
pair_positions = np.asarray(
	[slot_lookup[label] for label in ("Fp1h", "Fp2h", "AFF5h", "AFF6h")],
	dtype=np.float32,
)
ch_pos_1005 = np.repeat(pair_positions, 2, axis=0)
model = MtReNIRS.from_pretrained(
	ch_types=ch_types,
	ch_pos_1005=ch_pos_1005,
	device=device,
)

# Four already preprocessed trials: B=4, C=8, 20 seconds at 10 Hz.
trials = torch.randn(4, 8, 200, device=device)

window_embeddings = model.embed(trials)
trial_features = model.embed(trials, pool="flatten")

print("per-window embeddings (B, W, C, D):", tuple(window_embeddings.shape))
print("fixed-trial features (B, W*C*D):", tuple(trial_features.shape))
