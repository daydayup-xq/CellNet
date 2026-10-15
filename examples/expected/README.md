# Tested CPU outputs — CellNet 0.1.3

These outputs use the released Tabula Sapiens and TotalX PBMC fixtures, CPU FP32 SDPA, batch size 2 and four threads. See `metrics.json` and each task directory for predictions, representations and metrics. The model is frozen throughout.

- Annotation: 192 Tabula cells, six tissue-conditioned donor-LODO folds, native CLS + train-fitted StandardScaler + LDA. Equal-tissue accuracy 0.765625; Macro-F1 0.747464.
- Mapping: 96 PBMC cells, five stratified 80/20 splits, raw layer8 + reference scaler + k=10 cosine/prior-corrected KNN. Mean accuracy and Macro-F1 are 1.0 on this small four-class example.
- Integration: raw layer6, 90 neighbors, Leiden resolution 0.05. The small cohort forms one cluster (NMI and ARI both zero); the formal graph settings are retained. These scores do not estimate full-tissue integration performance.
- Completion: eight cells, five full gene-axis folds; all 36,499 genes are hidden once. 5,055 genes have finite PCC, with equal-gene mean PCC 0.265852. No positive-cell endpoint requiring 20 cells is reported.
- Association: contextual hidden-state means, panel centering and cosine similarities for four contexts; no biological ground-truth ranking is claimed.
- Denoising: extra API example outside Fig2–5. It does not outperform its noisy-input baseline; see its saved common-target metrics.

Protocol checks and validation scope are recorded in `docs/FIGURE_PROTOCOL_VALIDATION.json`.
