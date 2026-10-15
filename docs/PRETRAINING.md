# Pretraining CellNet

`CellNetPretrainer` trains aligned views of the same cell with global-view mean alignment and SIGReg (17 integration knots, 1,024 random slices, regularization weight 0.05). A supplied expression-target batch adds the p=1.5 compound Poisson–Exponential likelihood with inverse-sampling correction. The projection head is 768→1024→1024→256.

## Initialize from DINOv3

Download **DINOv3 ViT-B/16, LVD-1689M**, from [Meta's Hugging Face repository](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). Request access on its model page, then authenticate and download the safetensors file:

```bash
pip install -e '.[pretraining]'
hf auth login
hf download facebook/dinov3-vitb16-pretrain-lvd1689m model.safetensors \
  --local-dir ../pretrained_models/dinov3-vitb16
python examples/prepare_pretraining_batch.py --output prepared_views.pt
python examples/pretrain.py --release . --init dinov3 \
  --dinov3-weights ../pretrained_models/dinov3-vitb16/model.safetensors \
  --batch prepared_views.pt --n-global 2 --steps 1 --device cpu \
  --output trained_weights
```

The loader also accepts the [timm ViT-B/16 checkpoint](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m): use `hf download timm/vit_base_patch16_dinov3.lvd1689m model.safetensors --local-dir ../pretrained_models/dinov3-vitb16-timm` and pass that file to `--dinov3-weights`. Other model sizes and QKV-bias variants are not interchangeable with this architecture. Downloads follow the upstream DINOv3 license and access conditions.

Initialization loads all 12 Transformer blocks, final normalization, CLS and register tokens. The loader maps LayerScale names and, for Meta's Transformers format, concatenates Q/K/V projection weights. QKV biases must be zero, matching CellNet's bias-free attention. RGB patch projections and the unused mask token are excluded. Missing backbone tensors, unknown tensors and shape mismatches raise an error before the backbone is modified.

The CellNet expression/gene-identity input stem, decoder and projection head start with fresh parameters. The frozen gene-identity table is read from `weights/cellnet.pt` together with the configuration and vocabulary assets; trained CellNet encoder/decoder/projector parameters are not used in this mode. The spatial rotary embeddings use CellNet's input grid and timm implementation. DINOv3 files remain outside the project and are never downloaded implicitly by the training script.

## Continue from CellNet

Use `--init cellnet` (the default) to load both `weights/cellnet.pt` and `weights/projector.pt`:

```bash
python examples/pretrain.py --release . --init cellnet --batch prepared_views.pt \
  --n-global 2 --steps 10 --device cpu --output continued_weights
```

The command runs a single-process forward/backward/update/save example on prepared batches. It repeats the supplied batch; use independent batches and a data loader for a training run. The full-scale training schedule is not included. The optional expression loss trains expression prediction; a separate confidence-calibration objective is not implemented in this example.

## Batch format

Save a tensor-only dictionary with `torch.save`:

- `image`: float tensor `[batch, views, 2, 384, 384]`, channels expression in [0,1] and visibility mask.
- `gene_ids`: integer tensor `[batch, views, 384, 384]`; pretrained vocabulary index plus one, with zero reserved for unavailable/hidden genes.
- View index must refer to the same cell across all views. Place global views first and choose `--n-global` accordingly. Create independent masks/augmentations before batching; repeating identical views is only a smoke test.
- `batch >= 2` is required for projection-head batch normalization.

Optional `targets` dictionary:

- `gene_ids`: `[batch, targets]` integer query IDs.
- `mask`: boolean tensor of the same shape; false entries are padding.
- `expression_normalized`: separate ground-truth expression values in [0,1], same shape.
- `positive_pool_count`, `zero_pool_count`: `[batch]` sizes of each cell's complete candidate pools, before balanced subsampling. Positives use threshold 0.03 on the normalized scale. Populate these honestly: sampled counts cannot replace full pool counts.

All queried targets are removed from every input view before encoding. The example decodes from the first global view. Never recompute image normalization using hidden truths. The stem retains pair-preserving jitter within 16×16 patches during training.

```bash
python examples/pretrain.py --release . --init cellnet --batch prepared_views.pt \
  --n-global 2 --steps 10 --device cpu --output continued_weights
```

For real training use an appropriate GPU and independent batches through a custom data loader. The example repeats the supplied prepared batch and is intended to demonstrate the full forward/backward/update/save path. It uses AdamW (learning rate 2e-5, weight decay 0.05) and gradient clipping at 1.0. Exported tensors contain no optimizer or trainer state.

To use the continued weights, create a new release directory with the same `assets/` and place the three output files in its `weights/` directory. Keep any new run records separately from the pure model tensor files.

## Real bundled training data

See [datasets/pretraining/README.md](../datasets/pretraining/README.md) for 24 actual training cells, format and loader examples, independent masked-view preparation, visualizations, and a clearly marked placeholder for the future full dataset.
