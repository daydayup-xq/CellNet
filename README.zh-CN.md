# CellNet

CellNet 是一个融合基因身份与表达信息的单细胞基础模型。它将细胞表达谱转换为二维图像，提供细胞表征与基因表达预测，支持细胞类型注释、参考映射、批次整合及基因关联分析。

完整接口和命令见 [README.md](README.md)。

## 包含的内容

- 核心模型：基因身份与表达融合的 ViT-B、细胞表征提取、表达与置信度解码器。
- 纯权重：`weights/cellnet.pt`；预训练投影头单独保存为 `weights/projector.pt`。
- 最小配置：仅保留实际使用的模型维度、输入范围和表征参数。
- 已拟合的构图资产：约百万基因的词表、40 个物种的原生布局，以及支持新基因投影的 PCA、t-SNE 坐标与检索索引。
- 精简预训练代码、H5AD 推理示例、测试和完整性检查。

项目提供模型权重、基因词表、拟合好的图像转换器、预训练工具及真实数据使用示例。

## 真实数据多任务示例

[任务示例说明](docs/TASK_EXAMPLES.md) 配套 Tabula Sapiens 的 192 个真实细胞和 TotalX PBMC 的 96 个真实细胞。注释使用组织内留一供体验证及原生特征 + StandardScaler + LDA；参考映射、批次整合、五折基因补全和基因关联使用[实验对应参数](docs/FIGURE_PROTOCOLS.md)。去噪作为 Fig2–Fig5 之外的额外接口示例。

```bash
pip install -e '.[examples]'
python examples/run_all_tasks.py --device cpu --batch-size 2 --output example_outputs
```

各任务可单独运行，文件位于 `examples/`；数据位于 `examples/data/`，测试输出位于 `examples/expected/`。任务结果与验证范围见 [真实数据验证](docs/FIGURE_PROTOCOL_VALIDATION.json)。这些是 API 演示，不是全量论文基准。

构图时将表达值存为 float16，再转换为 float32 模型输入。五折补全输入中整折目标基因列均物理缺失，归一化只看可见计数，真值单独保存。

## 预训练数据示例与完整数据预留位置

[预训练数据目录](datasets/pretraining/README.md) 已包含从 10 万采样库提取的 24 个真实细胞：小鼠、斑马鱼、果蝇各 8 个。表达数组逐值保留原始 float16 数值，附测量掩码、基因 ID、独立多视图批次生成代码与可视化 PNG/SVG。

完整数据将放在 `datasets/pretraining/full/`，当前明确标记为尚未包含。示例位于 `example/`，两者使用相同格式和读取接口；来源、第三方许可核验状态与格式说明均随附。

## 安装与使用

先安装 [Git LFS](https://git-lfs.com/)，用于下载模型权重、构图资源和示例数据。Python 需为 3.10 或更新版本。

```bash
git lfs install
git clone https://github.com/daydayup-xq/CellNet.git
cd CellNet
git lfs pull
pip install -e '.[h5ad]'
python examples/embed_h5ad.py \
  --release . --input cells.h5ad --output embeddings.npy \
  --species Human --input-scale counts --device cpu
```

基因标识使用物种对应的稳定 ID。已知物种直接应用原生布局，不需要重新拟合。`counts` 会按总量 10,000 归一化、取 `log1p`，再裁剪到 [0,6] 并除以 6。已完成相同归一化的数据使用 `log1p`。

测到的零表达基因保留测量掩码和基因身份；未测量基因同时去掉表达、测量掩码和身份。基因补全时，应先移除隐藏目标再做输入归一化，并将真值单独保管；否则总量归一化本身也可能泄露隐藏目标的信息。

原生表征是最后八层原始 CLS 相加后再做 LayerNorm。`--layer layer8` 和 `--layer layer6` 分别表示倒数第八层和第六层的原始 CLS；`--layer layer8_normalized` 表示倒数第八层单独归一化后的 CLS，`--layer last` 表示最后一层归一化后的 CLS。复现特定下游协议时应明确选择，不能混用。

## 新基因与新物种

安装 `pip install -e '.[projection]'` 后，可用 `CellNetImageConverter.from_embeddings(...)` 应用原拟合器，并为当前模型追加冻结的基因身份行。已有基因保留原坐标和身份，新的位置使用原始均衡像素分配算法解决冲突。

输入需要与预训练身份相同特征空间的原始 1,920 维基因向量。不同模型生成的任意向量不能直接替代；原始向量的生成不在此发布包范围内。默认不静默忽略未知基因。

## 训练与验证

预训练支持从 [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) 初始化，也支持从 CellNet 权重继续训练。请先在 Hugging Face 模型页面申请访问权限，再下载权重：

```bash
pip install -e '.[pretraining]'
hf auth login
hf download facebook/dinov3-vitb16-pretrain-lvd1689m model.safetensors \
  --local-dir ../pretrained_models/dinov3-vitb16
python examples/prepare_pretraining_batch.py --output prepared_views.pt
python examples/pretrain.py --release . --init dinov3 \
  --dinov3-weights ../pretrained_models/dinov3-vitb16/model.safetensors \
  --batch prepared_views.pt --steps 1 --device cpu --output trained_weights
```

DINOv3 文件由用户单独下载，项目中不附带该权重。加载时迁移 Transformer 骨干、CLS 和 register tokens，重新初始化 CellNet 输入层、解码器及投影头，并使用项目提供的冻结基因身份表。已有 CellNet 模型的推理不需要另外下载 DINOv3。完整参数、兼容的 timm 权重入口和继续训练方法见[预训练说明](docs/PRETRAINING.md)。

```bash
python -m unittest discover -s tests -v
python examples/verify_release.py --release .
```

权重文件的 SHA256 见 `SHA256SUMS`。任务与训练验证分别见 `docs/FIGURE_PROTOCOL_VALIDATION.json` 和 `docs/DINOV3_INITIALIZATION_VALIDATION.json`。

## Fig2–Fig5 数据集来源

下表列出主图及相关补充分析对应的研究名称和官方入口。完整数据请从原研究或官方数据仓库获取。

| 图号 | 研究 / 数据集 | 官方链接 |
| --- | --- | --- |
| 2 | Tabula Sapiens 2.0 — human multi-organ atlas | [Figshare](https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984) |
| 2 | An atlas of healthy and injured cell states and niches in the human kidney | [Study](https://www.nature.com/articles/s41586-023-05769-3) · [CELLxGENE](https://cellxgene.cziscience.com/collections/bcb61471-2a44-4d00-a0af-ff085512674c) |
| 2 | A molecular cell atlas of the human lung from single-cell RNA sequencing (Travaglini et al.) | [Study](https://www.nature.com/articles/s41586-020-2922-4) · [CELLxGENE](https://cellxgene.cziscience.com/collections/5d445965-6f1a-4b68-ba3a-b8f765155d3a) |
| 2, 4 | Single-cell transcriptomics identifies an effectorness gradient shaping the response of CD4+ T cells to cytokines (Cano-Gamez et al.) | [Study](https://www.nature.com/articles/s41467-020-15543-y) · [BioStudies S-BSST2978](https://www.ebi.ac.uk/biostudies/studies/S-BSST2978) |
| 2 | Single-cell transcriptomic analysis uncovers diverse and dynamic senescent cell populations (WI-38) | [Study](https://www.aging-us.com/article/204666/text) · [GSE226225](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226225) |
| 2 | Seven-species circulating immune-cell atlas (author H5ad collection) | [Figshare 28008611](https://doi.org/10.6084/m9.figshare.28008611) |
| 2, 3 | The molecular evolution of spermatogenesis across mammals | [Study and data availability](https://www.nature.com/articles/s41586-022-05547-7) · [Author portal](https://apps.kaessmannlab.org/SpermEvol/) |
| 2 | Recruitment, rewiring, and deep conservation in flowering plant gene regulation (Brassicaceae multi-tissue atlas) | [GSE297576](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE297576) |
| 2 | Evolutionary diversity of cell-type-specific expression and stress response in Brassicaceae roots | [GSE268881](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268881) |
| 3, 4, 5 | Scalable single-cell total RNA sequencing unifies coding and noncoding transcriptomics (TotalX: PBMC, developing brain, HEK293T and Huh7, including DENV conditions) | [Study](https://www.nature.com/articles/s41587-026-03068-6) · [Processed matrices](https://doi.org/10.6084/m9.figshare.29803628) · [Cell-line data GSE315939](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE315939) |
| 3 | Xenium Prime: Fresh Frozen Mouse Brain Hemisphere with 5K Mouse Pan Tissue and Pathways Panel | [10x Genomics dataset](https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain) |
| 5 | Massively parallel digital transcriptional profiling of single cells (Zheng et al.; purified PBMC) | [Study](https://www.nature.com/articles/ncomms14049) · [Processed data loader](https://docs.scvi-tools.org/en/1.5.0/api/reference/scvi.data.purified_pbmc_dataset.html) |
| 5 | Human Brain Cell Atlas v1.0 — perirhinal cortex | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443) · [scGPT processed data](https://github.com/bowang-lab/scGPT/blob/main/data/README.md) |
| 5 | High-throughput total RNA sequencing in single cells using VASA-seq | [Study](https://www.nature.com/articles/s41587-022-01361-8) · [GSE176588](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176588) |
| 5 | Dual genome-wide coding and lncRNA screens in neural induction of induced pluripotent stem cells | [Study](https://doi.org/10.1016/j.xgen.2022.100177) · [GSE150062](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150062) |
