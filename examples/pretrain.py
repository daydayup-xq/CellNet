"""Train CellNet from DINOv3 initialization or continue pretrained CellNet."""

import argparse
from pathlib import Path
import torch
from cellnet.pretraining import CellNetPretrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Tensor-only .pt file; see docs/PRETRAINING.md",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-global", type=int, default=2)
    p.add_argument("--init", choices=["cellnet", "dinov3"], default="cellnet",
                   help="Continue CellNet, or initialize a new model with DINOv3")
    p.add_argument("--dinov3-weights", type=Path,
                   help="User-downloaded Hugging Face DINOv3 ViT-B/16 model.safetensors")
    args = p.parse_args()
    if args.init == "dinov3" and args.dinov3_weights is None:
        p.error("--init dinov3 requires --dinov3-weights; see docs/PRETRAINING.md")
    if args.init == "cellnet" and args.dinov3_weights is not None:
        p.error("--dinov3-weights requires --init dinov3")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    batch = torch.load(args.batch, map_location="cpu", weights_only=True)
    image = batch["image"].to(args.device)
    ids = batch["gene_ids"].to(args.device)
    targets = batch.get("targets")
    if args.init == "dinov3":
        trainer = CellNetPretrainer.from_dinov3(
            args.release, args.dinov3_weights, args.device).train()
        print(trainer.initialization_report, flush=True)
    else:
        trainer = CellNetPretrainer.from_pretrained(args.release, args.device).train()
    optimizer = torch.optim.AdamW(
        (p for p in trainer.parameters() if p.requires_grad), lr=2e-5, weight_decay=0.05
    )
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        result = trainer(image, ids, n_global=args.n_global, targets=targets)
        if not torch.isfinite(result["loss"]):
            raise RuntimeError("Nonfinite training loss")
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
        optimizer.step()
        print(
            step + 1,
            {k: round(float(v.detach()), 6) for k, v in result.items()},
            flush=True,
        )
    args.output.mkdir(parents=True)
    # Save pure tensors, without optimizers, callbacks or machine-specific configuration.
    torch.save(trainer.model.state_dict(), args.output / "cellnet.pt")
    torch.save(trainer.projector.state_dict(), args.output / "projector.pt")
    import json

    config = dict(trainer.model.config, model_name="CellNet")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
