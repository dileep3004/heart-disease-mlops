# Dataset

The code downloads the official UCI Heart Disease archive on demand and extracts the
processed Cleveland subset (`processed.cleveland.data`). The assignment uses 13 features
and a binary target. The source `num` field has values 0..4; this project maps `0 -> 0`
and `1..4 -> 1`.

Run `python -m heart_mlops.data --output data/processed/heart_disease_clean.csv` after
installing the package. The script stores the raw file, cleaned CSV, and a JSON provenance
manifest containing the data URL, SHA-256 hash, row count, class counts, and missing-cell
count. Missing predictor values remain NaN in the cleaned CSV and are imputed inside the
scikit-learn training pipeline to prevent train/test leakage.

Dataset citation: Janosi et al. (1989), UCI Machine Learning Repository, DOI
https://doi.org/10.24432/C52P4X. Dataset licence listed by UCI: CC BY 4.0. Cite the source
and follow the repository's attribution terms if redistributing the dataset.
