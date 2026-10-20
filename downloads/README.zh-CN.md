# CellNet Image Atlas

[下载完整 Image Atlas](https://pan.baidu.com/s/1FuoHnEJN9FRvOBio5uutYQ?pwd=xabe) · 提取码：**xabe**。

面向 CellNet 的单细胞转录组图像集合，同时提供基因身份词表、测量掩码和已拟合的像素映射。源元数据记录 231,190,557 个细胞表达谱、26,467 个来源数据集、40 个物种与 1,015,163 个基因身份。

## 下载到哪里

先克隆 [CellNet](https://github.com/daydayup-xq/CellNet)，通过 Git LFS 获取模型资产。将**本下载目录里面的内容**直接放入 `CellNet/datasets/pretraining/full/`：

```text
CellNet/                         # --release 指向这里
├── assets/                      # 已发布的词表与拟合布局
├── weights/                     # 编码器、DNA 基因身份与解码器
└── datasets/pretraining/full/
    ├── README.md
    ├── README.zh-CN.md
    ├── DATASET_CARD.json
    ├── metadata.json
    ├── gene_vocab.json
    ├── cell_type_vocab.json
    ├── cell_type_storage_vocab.json
    ├── normalization_stats.json
    ├── noncoding_gene_ids.json
    ├── release_manifest.json
    ├── shards_manifest.json.gz
    ├── cell_sources.json.gz
    ├── dataset_sources.tsv
    ├── class_index.json
    ├── class_index.tsv
    ├── class_shards/class_*/shard-*.tar.gz
    ├── masks/
    │   ├── dataset_index.json
    │   ├── gene_coords.npy
    │   └── gene_masks.npy
    └── pixel_maps/
        ├── metadata.json
        ├── dataset_index.json
        ├── dataset_to_pixel_map.json
        ├── global_gene_coords.npy
        └── gene_to_flat_pixel.npy
```

确保存在 `full/metadata.json`，不要多套一层下载目录。保留所有相对路径与文件名。分片可能为 `.tar` 或 `.tar.gz`，**无需解压分片**，读取器会直接流式读取；如果下载中包含 `class_shards/unlabeled/`，也应保留。`release_manifest.json` 列出配套资产，并引用 `shards_manifest.json.gz` 中全部 512,835 个原生分片的压缩清单；下载时请同时保留这两份清单。

## 预训练兼容性

请将 `cell_sources.json.gz` 与 `metadata.json` 放在同一目录。这个紧凑索引记录了 **189 个来源数据集、44,489,429 个合并细胞**的逐细胞来源。原生读取器会根据 `release_manifest.json` 校验索引，并自动选用每个细胞对应的测量掩码。

预训练复用已发布的基因到图像映射。下游应用使用配套模型的 `assets/` 与 `weights/` 处理您的 H5AD，无需下载完整 Image Atlas。

## 核验与预训练示例

设置克隆仓库的绝对路径后，可从任意目录执行：

```bash
export CELLNET_RELEASE="/absolute/path/to/CellNet"
git -C "$CELLNET_RELEASE" lfs pull
python -m pip install -e "$CELLNET_RELEASE[h5ad]"

python "$CELLNET_RELEASE/examples/verify_image_atlas.py" \
  --release "$CELLNET_RELEASE" \
  --data "$CELLNET_RELEASE/datasets/pretraining/full"

python "$CELLNET_RELEASE/examples/prepare_pretraining_batch.py" \
  --release "$CELLNET_RELEASE" \
  --data "$CELLNET_RELEASE/datasets/pretraining/full" \
  --output "$CELLNET_RELEASE/prepared_views.pt" \
  --cells 2 --views 2

python "$CELLNET_RELEASE/examples/pretrain.py" \
  --release "$CELLNET_RELEASE" --init cellnet \
  --batch "$CELLNET_RELEASE/prepared_views.pt" \
  --n-global 2 --steps 1 --device cpu \
  --output "$CELLNET_RELEASE/continued_weights"
```

上面演示用一个小批次完成一次模型更新。正式训练应持续读取批次，并选择合适的设备与训练配置，详见仓库[预训练说明](https://github.com/daydayup-xq/CellNet/blob/main/docs/PRETRAINING.md)。批次准备工具在省略 `--data` 时，优先使用 `--release` 下已安装的 `datasets/pretraining/full/`，否则使用独立的 24 细胞示例。

核验工具检查发布清单及文件大小、词表与拟合映射的一致性，并实际读取少量细胞。添加 `--checksums` 会读取清单内文件并比较已发布且有核验依据的摘要；具体覆盖范围以输出报告为准。抽样读取不会被视为全体细胞计数。

## 预训练与下游自动复用同一套映射

两种流程都从 `--release` 加载**已经拟合好的**资产。原生数据读取器将图谱词表、物种像素位置与 `assets/gene_vocab.json`、`assets/species_layouts.npz` 核对；下游转换器也自动复用同一套固定位置。请保持 `assets/` 和 `weights/` 来自同一模型版本，**无需重新拟合布局**。

使用自己的 H5AD 做下游应用时，可以直接使用模型资产，不必先下载完整细胞集合：

```bash
python "$CELLNET_RELEASE/examples/embed_h5ad.py" \
  --release "$CELLNET_RELEASE" \
  --input "/path/to/cells.h5ad" --output "/path/to/embeddings.npy" \
  --species Human --input-scale counts --batch-size 8 --device cpu
```

物种名称须与拟合命名空间完全一致，基因使用稳定 ID。`counts` 表示非负计数；`log1p` 表示按模型要求归一化至 10,000 文库大小再 log1p 的表达值。基因表达预测、基因关联等应用见[下游示例](https://github.com/daydayup-xq/CellNet/blob/main/docs/TASK_EXAMPLES.md)。

DNA 衍生的冻结 **256 维基因身份向量**已包含在 `weights/cellnet.pt` 中，并与发布词表对齐。本 Image Atlas 提供表达图像、掩码与映射，不包含原始 1,920 维 DNA 嵌入矩阵。已收录基因会自动使用冻结身份向量；注册新基因需提供兼容的原始嵌入，并使用仓库说明的可选拟合投影资产。

## 数据格式与来源

存储表达为 float16，范围 [0,1]，已经归一化，请勿重复归一化。读取结果 `image` 为 float32 `[2,384,384]`，分别对应表达与测量掩码；`gene_ids` 为 int64 `[384,384]`。实测但表达为零的像素仍为 mask=1、基因 ID>0；未测量像素为表达=0、mask=0、ID=0。

`masks/dataset_index.json` 定义来源数据集的掩码行；`pixel_maps/dataset_index.json` 定义物种像素映射的行，两者不是同一索引。`dataset_sources.tsv` 在保持原索引不变的前提下列出数据集数字 ID、公开来源标识及物种。`cell_type_vocab.json` 定义生物学标签；`class_index.json` 与 `.tsv` 帮助定位存储桶，一个桶可能包含多个标签。桶字段为空表示历史索引未记录对应关系，原生读取器仍直接使用每个细胞记录中的标签。

上述细胞规模来自源元数据；发布清单记录本次文件核验范围。原数据集的署名与使用条款继续适用；模型仓库 Apache-2.0 许可适用于软件，不替代第三方数据许可。

## 先试用 24 个真实细胞

`CellNet_pretraining_example_24cells.zip` 是独立的小型示例包：小鼠、斑马鱼和果蝇各 8 个细胞，包含表达、测量掩码、基因 ID、词表和拟合布局。解压后使用 `CellNet/datasets/pretraining/example/`。它与上方网盘中的完整原生图谱分开提供。

本网页的 `SHA256SUMS` 与 `FILE_CATALOG.json` 覆盖示例 ZIP 和网页配套说明。完整图谱的逐文件清单在网盘包内的 `release_manifest.json` 和 `shards_manifest.json.gz`。
