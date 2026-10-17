# Release validation — 17 October 2026

The source research directory was not modified. Packaging created a separate `ArchaEC-SVM` directory and excluded cached environments, checkpoints, scratch MMseqs2 databases, historical result archives, unpublished manuscripts, and the original smaller-dataset summaries.

Completed checks:

- 445 copied source assets matched their original SHA-256 hashes.
- All 11,425 accession records, 10,854 unique sequences, 204 taxonomy IDs and seven overlapping label counts were checked.
- Every sequence hash matched the source sequence.
- All 11,425 rows of the 432-dimensional features were recomputed with the portable implementation and exactly matched the frozen matrix.
- 55 main SVM outer test folds were checked for partition/label alignment, coverage and metric agreement. Five-fold means and sample SD were recomputed.
- An additional 70 algorithm-comparison/feature-ablation fold AUC/AP calculations matched the stored summaries.
- Six unit tests passed, including parity against the original feature, mRMR and threshold functions; invalid input and row-order rejection were tested.
- The synthetic CPU installation test passed.
- A fresh AAC-only SVM training run on random fold 0 reproduced all seven original test scores exactly for all 2,285 test accessions (maximum absolute difference 0.0 in the local recorded environment).
- Checkpoint resume and prediction CSV export passed on that new run.
- The verification-plot entry point produced separate PNG/SVG/PDF plots without changing the released publication assets.
- Public-source file size checks found no individual file at or above 90 MiB; the largest file is the 19,742,528-byte feature matrix.
- Scans for GitHub tokens, AWS-like keys, private-key headers and absolute author-local paths found no confirmed credential or author-local path in the selected release. AWS-prefix-like hits were verified to be substrings of canonical protein sequences, not credentials.
- Local links in the authored documentation were checked.

The full experiment grid was **not retrained** during packaging. The fresh training parity check covers one AAC ablation fold; it does not independently reproduce every model or every explanation. A newly installed clean environment may exhibit version/platform-dependent numerical differences. The portable training and prediction commands are separate from the unchanged, workspace-dependent archival plotting scripts.

The source comparison/explanation record remains available in `original_run/execution_environment.json`. This release adds no claim of external validation, calibrated probabilities, or causal feature interpretation.
