"""DINOv3 ViT-B/16 initialization from user-downloaded Hugging Face weights."""
from pathlib import Path
import torch

DINOV3_REPOSITORY = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TIMM_REPOSITORY = "timm/vit_base_patch16_dinov3.lvd1689m"


def load_dinov3_backbone(backbone, checkpoint):
    """Load every non-stem encoder tensor; reject incompatible weights before mutation."""
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download DINOv3 model.safetensors first: {checkpoint}")
    raw = load_file(str(checkpoint), device="cpu")
    source, excluded, qkv = {}, [], {}
    aliases = {"embeddings.cls_token": "cls_token", "embeddings.register_tokens": "reg_token"}
    unused = {"patch_embed.proj.weight", "patch_embed.proj.bias",
              "embeddings.patch_embeddings.weight", "embeddings.patch_embeddings.bias",
              "embeddings.mask_token"}
    for name, value in raw.items():
        if name in unused:
            excluded.append(name)
            continue
        if name.startswith("layer.") and ".attention." in name:
            prefix, part = name.split(".attention.", 1)
            if part in {f"{p}_proj.bias" for p in ["q", "k", "v"]}:
                if torch.count_nonzero(value).item():
                    raise ValueError(f"CellNet requires zero DINOv3 QKV biases: {name}")
                excluded.append(name)
                continue
            if part in {f"{p}_proj.weight" for p in ["q", "k", "v"]}:
                qkv.setdefault(prefix.replace("layer.", "blocks."), {})[part[0]] = value
                continue
        key = aliases.get(name, name)
        for before, after in [("layer.", "blocks."), (".attention.o_proj.", ".attn.proj."),
                              (".mlp.up_proj.", ".mlp.fc1."), (".mlp.down_proj.", ".mlp.fc2."),
                              (".gamma_1", ".ls1.gamma"), (".gamma_2", ".ls2.gamma"),
                              (".layer_scale1.lambda1", ".ls1.gamma"),
                              (".layer_scale2.lambda1", ".ls2.gamma")]:
            key = key.replace(before, after)
        if key in source:
            raise ValueError(f"Duplicate DINOv3 parameter mapping: {key}")
        source[key] = value
    for prefix, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete QKV projections: {prefix}")
        key = prefix + ".attn.qkv.weight"
        if key in source:
            raise ValueError(f"Duplicate DINOv3 QKV parameter: {key}")
        source[key] = torch.cat([parts[p] for p in ["q", "k", "v"]], dim=0)
    target = backbone.state_dict()
    required = {key for key in target if not key.startswith("patch_embed.")}
    missing = sorted(required - source.keys())
    unexpected = sorted(source.keys() - required)
    mismatched = sorted(key for key in required & source.keys()
                        if source[key].shape != target[key].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f"Expected DINOv3 ViT-B/16 safetensors from {DINOV3_REPOSITORY} "
                         f"or {TIMM_REPOSITORY}; missing={missing}, "
                         f"unexpected={unexpected}, shape_mismatch={mismatched}")
    backbone.load_state_dict(source, strict=False)
    return {"backbone_source": DINOV3_REPOSITORY if "embeddings.cls_token" in raw else TIMM_REPOSITORY,
            "loaded_tensors": len(required), "excluded_source_tensors": sorted(excluded)}
