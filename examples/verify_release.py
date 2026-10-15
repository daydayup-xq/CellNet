"""Verify delivered checksums, file independence and tensor-only weight contents."""

from pathlib import Path
import argparse
import hashlib
import torch


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    args = p.parse_args()
    root = args.release.resolve()
    files = list(root.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise RuntimeError("Release contains symbolic links")
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError("Invalid checksum path")
        if digest(target) != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
    for name in ["cellnet.pt", "projector.pt"]:
        state = torch.load(
            root / "weights" / name, map_location="cpu", weights_only=True, mmap=True
        )
        if not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise RuntimeError("Weights contain objects other than tensors")
        if any("species_" in k or "prototype" in k for k in state):
            raise RuntimeError("Weights contain excluded training state")
        print(name, len(state), "tensors")
    print("Checksums and release structure verified")


if __name__ == "__main__":
    main()
