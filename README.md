# QE-MTS seeding controls: code and data

Code and data repository: https://github.com/QmuseAI/QE-MTS-seeding-controls

## Layout

- `step1_digitize/`: extraction code and numerical published-curve data; publisher PDF and overlays excluded.
- `step2_cluster/`: C solvers, Python analysis, SLURM configurations, raw data archives and seeds.
- `gpu_dcqo/`: independent DCQO simulation and diagnostic logs. Q1/Q5 seed strings are stored once in `step2_cluster/seeds/seed_files.tar.gz`.
- `analysis/frozen/`: final statistical tables, bootstrap draws and generation scripts moved out of manuscript revision history.
- `article/`: figure scripts only; manuscript drafts and revision history excluded.

## Reproduce figures

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python article/make_article_figs.py
python article/make_figA8_seed_worth.py
```

The scripts use included summary tables; no search experiments are rerun.

## Recompute final statistical intervals

```bash
python analysis/frozen/freeze_core_ci.py
python analysis/frozen/derive_secondary_intervals.py
```

This reads compressed raw records and overwrites frozen outputs. It uses 2,000 bootstrap draws with seed 20261102 and may take substantial time. The original input manifest is retained; the release script changes the project-root path and removes manuscript backup operations.

## Extract raw data and seeds

```bash
tar -xzf step2_cluster/run_main/data.tar.gz -C step2_cluster/run_main
tar -xzf step2_cluster/run_ext/data.tar.gz -C step2_cluster/run_ext
tar -xzf step2_cluster/seeds/seed_files.tar.gz -C step2_cluster/seeds
```

See component READMEs for historical cluster commands and GPU requirements. CUDA-Q and NVIDIA hardware are additional requirements for simulation. Adapt cluster account, partition and environment paths before submitting jobs. Component READMEs describe the historical workflow; final figure intervals use `analysis/frozen/` rather than older analysis outputs. Saved DCQO seeds are extracted into the cluster seeds directory, not the GPU output directory.

## Data and reproducibility

Raw rows contain `r s evals best_energy`, plus a configuration column in extension records. Negative evaluation counts indicate censored runs. The frozen analysis evaluates censoring bounds.

Historical provenance files are retained. Some main-run source hashes differ from current sources; complete historical binary provenance is not established. Packaging does not constitute a full rerun of the experiments.

DCQO seeds come from independent simulation. Quantum generation costs are not included in downstream search counts. These data do not establish an end-to-end quantum speedup.

## Publication

Code and data are bundled together at https://github.com/QmuseAI/QE-MTS-seeding-controls. No reuse license has been specified by the authors.

The source publication is Gomez Cadavid et al., Quantum Machine Intelligence 8, 92 (2026), DOI: 10.1007/s42484-026-00433-1. Obtain its PDF from the publisher to rerun digitisation.
