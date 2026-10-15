# Figure protocols and representative examples

The README lists study names and official data sources for Figures 2–5. Full figure cohorts are downloaded separately. The bundled examples retain the specified model readouts and downstream estimators while reducing cohort size; their scores are not figure reproductions.

| Task | Actual experimental settings | Released example |
| --- | --- | --- |
| Cell annotation | LayerNorm(sum of last eight raw CLS tokens); training-only StandardScaler; LDA(solver="lsqr", shrinkage="auto"), empirical class priors | 192 Tabula Sapiens cells; two tissues; within-tissue leave-one-donor-out; average folds, then tissues |
| Rare-cell / cell-state classification | Same native readout and LDA; donor partitions for kidney/lung; CD4 donor-LODO; WI-38 paired library-replicate folds | Estimator demonstrated by annotation; rare-cell eligibility and state cohorts are not rerun here |
| Within-species reference mapping | Raw CLS eight blocks from the end; reference-fitted StandardScaler; cosine distance-weighted KNN k=10; class-prior correction exponent 0.25; 80/20 stratified reference/query; seeds 42, 123, 2024, 2025, 20261008 | Five seeds on 96 real PBMC cells; fresh stratified splits with query cells excluded from reference |
| Cross-species reference mapping | Raw layer8; k=30; cosine distance weights; prior exponent 0.5; 50/25/25 reference/validation/test; a reference-fitted scaler per species | Documented separately; the within-species example must not be reported as this protocol |
| Batch integration | Raw CLS six blocks from the end; Euclidean graph, 90 neighbors; Leiden resolution 0.05, seed 0; scaled ASW | 96 PBMC cells. avg_bio=(NMI+ARI+cell-type ASW)/3; avg_batch=(batch ASW+graph connectivity)/2. Full figure averages tissue scores equally |
| Frozen gene completion | Five seed-shuffled gene-axis folds, seed 42; whole hidden fold excluded from expression, measurement mask and gene IDs; observed-only input normalization; separate scorer truth | Eight PBMC cells, all 36,499 supported genes hidden exactly once across five folds; native expected expression p*M; gene-wise PCC |
| Contextual gene association | Final decoder 512-D interaction states; mean over cells within context, center over panel genes, L2-normalize, cosine; up to 100 cells per context | 24 cells per type from the 96-cell PBMC fixture; 64-gene panel. Centering and similarities depend on panel composition |

For Fig3, full PBMC/brain cohorts contain 5,000 query cells. Positive-intensity evaluation uses the native positive magnitude M on at least 20 measured-positive cells per gene, rather than expected expression p*M; the eight-cell example cannot estimate that endpoint. Detection-rate filtering and 2,000-resample confidence intervals belong to the full benchmark, not the small-cohort point estimates. Fine-tuning is a separate protocol (30 epochs at the stated reference-cohort sizes), not an option silently applied to the frozen examples.

Fig4 tracing, its 3,279-gene network panel, and Fig5 module-recovery panels have their own cohort and panel definitions. The contextual similarity example demonstrates the aggregation operation; it does not claim to rerun attribution, perturbation controls or module-recovery benchmarks.

The additional `denoising.py` example uses binomial thinning of real counts (retention 0.5, seed 20261016). It is an API demonstration outside Figures 2–5. Its scores must not be mixed with OpenProblems or publication benchmark metrics.

All images use the float16 storage round-trip, then float32 inference. `embed_h5ad.py --layer native` exports annotation features; `--layer layer8` and `--layer layer6` export the raw mapping and integration features. `--layer layer8_normalized` explicitly selects the individually normalized representation.
