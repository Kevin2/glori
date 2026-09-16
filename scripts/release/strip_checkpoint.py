"""
Strip a Lightning checkpoint down to only what's needed for inference
(model weights + hyperparameters), dropping optimizer state, LR scheduler
state, and training loop bookkeeping. Used to shrink checkpoints before
public release (Hugging Face / Zenodo).

Usage:
    python strip_checkpoint.py <input.ckpt> <output.ckpt>
"""

import sys
from pathlib import Path

import torch

KEEP_KEYS = [
    "state_dict",
    "hyper_parameters",
    "hparams_name",
    "pytorch-lightning_version",
    "epoch",
    "global_step",
]


def strip_checkpoint(in_path: Path, out_path: Path) -> None:
    ckpt = torch.load(in_path, map_location="cpu")
    stripped = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}

    dropped = set(ckpt.keys()) - set(stripped.keys())
    print(f"Dropped keys: {sorted(dropped)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, out_path)

    in_size = in_path.stat().st_size / 1e6
    out_size = out_path.stat().st_size / 1e6
    print(f"{in_path.name}: {in_size:.1f} MB -> {out_size:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    strip_checkpoint(Path(sys.argv[1]), Path(sys.argv[2]))
