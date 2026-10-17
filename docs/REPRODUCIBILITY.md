# Reproducibility and release boundaries

## Frozen scientific evidence

The authoritative source run is `svm_migration_20261008`. All current copied tables, OOF predictions and figures have byte-level source hashes in `source_file_manifest.json`. The current model is RBF-SVM; other algorithms remain in Figure 3 as comparators. Historical single-split LightGBM values must not be mixed with current five-fold SVM results.

Figure 2 is a descriptive dataset figure, not a prediction experiment. The old `dataset_summary.csv` in the research workspace describes the initial smaller collection, so it is deliberately excluded. Use `updated_dataset_summary*.csv` for the Swiss-Prot dataset. The 3.4 directory retains its historical name but now contains the Figure 5 feature ablation experiment. Figure 6B is the later random-five-fold, lower-triangle revision, not the earlier Homology-40 matrix or the alternative dotplot preview.

## Portable workflow versus historical source

`model.py`, `train.py`, `train_5fold.py`, `predict.py`, and `scripts/validate_release.py` are release entry points. They use the frozen data and folds, preserve the feature formulas, train-only mRMR, IFS grid, SVM parameters, scaler dtype, threshold search and metric definitions of the source experiment. Original class-by-class SVM checkpoints are resumable; configuration fingerprints prevent incompatible reuse. A newly trained portable run is a new run, not a replacement of the published evidence.

The original 01–43 numbered scripts are retained unchanged under `archive/scripts/` for method and figure provenance. Some scripts require original relative directories, prior completed jobs, Windows process management, WSL2/MMseqs2, template images, plot revision archives, or local SHAP installations. They are **not advertised as clean-clone commands**. Archived comparison code is imported by `train.py` only for non-SVM algorithms, with its output directory and fold seed redirected to the portable run.

The released figures are final exported SVG/PDF/PNG assets. This release preserves their source but does not claim a portable single-command regeneration of every publication panel. The included `scripts/plot_released_results.py` creates clearly named verification plots from final numerical tables, without overwriting published figures. Original full model checkpoints and scratch histories are not bundled; training regenerates selected checkpoints.

## Interpretation

The released KernelSHAP arrays and sampling manifests derive from the source model checkpoints. The historical trainer's `run_shap` function documents the exact computation. KernelSHAP is approximate and background-dependent. Its additivity check is not a convergence proof. No new SHAP estimates are substituted during packaging.

Figure 6B uses complete single-label outer-test proteins under random splitting and differences of logit-transformed class scores. The masked diagonal has no binary classification interpretation. Feature group sizes differ, and correlated composition features prevent interpreting summed SHAP values as unbiased biological importance.

## Statistical limitations

The dataset was inspected during earlier development. Final SVM selection followed previous algorithm comparisons; there is no independent untouched external test set. Outer five-fold estimates are internal evaluations and training partitions overlap. Sample SD is neither a standard error nor a 95% confidence interval. Small mean differences do not establish significance. Class imbalance, incomplete labels, limited taxonomic coverage, and disparate comparator input representations remain limitations.

Homology thresholds are clustering parameters, not verified maximum all-pairs cross-fold similarities. The identical 10%/20% assignments are disclosed. AUC below 0.8 under low-homology conditions is retained as measured, not modified to meet a target.
