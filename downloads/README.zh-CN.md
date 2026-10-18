# CellNet Image Atlas

## Image Atlas

完整 CellNet Image Atlas 已开放，可通过[百度网盘](https://pan.baidu.com/s/1J9rDKjT3yEqelqt3kaSIeQ?pwd=xabe)获取。提取码：**xabe**。

源元数据记录的预训练集合包含 231,190,557 个细胞表达谱、26,467 个数据集、40 个物种和 1,015,163 个基因身份。`FULL_DATA_STATUS.json` 提供完整集合的访问方式与可用状态。

## 24 个真实预训练示例

`CellNet_pretraining_example_24cells.zip` 为可直接读取的真实预训练示例：小鼠、斑马鱼、果蝇各 8 个细胞，共 24 个，包含原始存储表达数组、数据集特异的实测基因掩码、原生基因 ID、共享基因词表、拟合布局、格式说明和图片。公开样本 ID 已匿名化；保留物种、细胞类型和来源数据集信息。

安装 CellNet 并将示例包解压至代码所在位置后，可使用：

```python
from cellnet.pretraining_data import iter_pretraining_cells
for cell in iter_pretraining_cells('CellNet/datasets/pretraining/example'):
    image, gene_ids = cell['image'], cell['gene_ids']
```

表达已归一化到 [0,1]，请勿重复归一化。实测但为零的像素仍为 mask=1；未测量像素为 mask=0、gene ID=0。图片由真实转录组表达计算渲染，便于查看表达纹理。示例可用于格式检查和数据读取演示。

## 文件与元数据

`SHA256SUMS` 提供本网站示例压缩包及配套元数据的校验值；`FILE_CATALOG.json` 列出这些文件以及示例压缩包中的九个成员。完整 Image Atlas 通过上方百度网盘链接访问。

示例包按数据示例、基因词表、拟合布局、预览图与发布元数据组织。`DATASET_CARD.json` 说明示例范围和完整集合的访问方式。

## 来源与使用条款

原数据许可与署名要求继续适用；软件的 Apache-2.0 不自动覆盖第三方数据，逐来源数据许可核验仍待完成。
