# Released data

The data are accession-level records, not a deduplicated set of 10,854 rows. The master table has 11,425 rows and 10,854 distinct sequences. Exactly identical sequences remain together across training/validation/test partitions. Multilabel EC1–EC7 counts overlap.

## Files

- `raw/uniprot_swissprot_archaea_ec_20260719.tsv`: byte-identical downloaded source table.
- `raw/source_manifest.json`: query, requested fields, release header, retrieval time and source SHA-256; only the original local path has been made relative.
- `processed/archaea_swissprot_ec_master.csv`: annotated, labeled master table.
- `processed/archaea_swissprot_ec_master.fasta`: accession-matched sequences.
- `features.npy`: frozen `(11425, 432)` float32 feature matrix, in master-row order.
- `features.accessions.txt`: one accession per feature row.
- `label_mapping.csv`: seven top-level EC labels and overlapping positive counts.
- `splits/*.csv`: frozen groups, outer-fold IDs and train/validation/test roles for each of five folds.
- `clusters/identity_*.tsv`: original MMseqs2 representative/member pairs. FASTA identifiers are retained exactly as in the source mapping. Group unions additionally protect identical sequences.

## Master-table dictionary

| Columns | Meaning |
| --- | --- |
| `accession`, `entry_name` | UniProt accession and entry name |
| `protein_names`, `organism`, `taxonomy_id`, `lineage` | source protein and taxonomic metadata |
| `sequence`, `length`, `sequence_sha256` | amino-acid sequence, length and sequence hash |
| `ec_numbers_raw`, `ec_numbers`, `top_classes`, `top_class_names` | original/normalized EC annotations and their first-level labels |
| `label_ec1` … `label_ec7` | binary top-level labels; multiple ones are allowed |
| `n_full_ec`, `n_top_classes` | numbers of distinct EC annotations and top-level classes |
| `is_multifunctional_full_ec`, `is_multifunctional_top_level` | flags derived from the two respective counts; they are not interchangeable |
| `protein_existence`, `fragment`, `active_site`, `binding_site`, `transmembrane` | source evidence/annotation fields; missing values remain missing |
| `phylum`, `class_taxonomy`, `genus` | derived taxonomic descriptors |
| `date_created`, `date_modified` | source record dates |

Reviewed annotations do not imply experimental confirmation of every EC label. Unannotated functions may be missing positives. Fragment entries remain in the training dataset; the specified comparative-interpretation populations exclude fragments and/or multilabel proteins. None of these data provide non-enzyme negatives.

## Feature convention

Columns 0–19: AAC in `ACDEFGHIKLMNPQRSTVWY` order. Columns 20–419: DPC in lexicographic pair order using the same alphabet. Columns 420–431: log(1+sequence length), residue entropy, longest hydrophobic run divided by canonical length, hydrophobic/charged/polar/aromatic/small/G/P/C/H fractions. `model.FEATURES` is the authoritative ordered list.

AAC denominators count canonical residues; DPC denominators count adjacent pairs with both residues canonical. Noncanonical residues do not bridge adjacent pairs. Physicochemical groups and exact denominators are preserved in `model.py`.

## Split convention

`outer_fold` ranges from 0 to 4. `role_fold_0` … `role_fold_4` assign `train`, `validation` or `test` for each evaluation. Every accession is an outer test sample exactly once. Identical-sequence union constraints can merge nominal species/cluster groups. Do not reshuffle the master rows without also rebuilding the aligned feature and split inputs.

The 10% and 20% files are intentionally identical. Use the released maps for the reported benchmark, rather than assuming a new MMseqs2 version or live database download will recreate the exact assignments.
