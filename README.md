# HGT Spatial Dependence — NCD Mortality in East Africa

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SallySims/HGT_GBD-Mortality/blob/main/GBDTransformers_FullRevised.ipynb)

## Overview

This repository contains the implementation code for a Heterogeneous Graph Transformer (HGT) model examining spatial predictive patterns in cause-specific NCD mortality across five East African countries (Burundi, Kenya, Rwanda, Tanzania, Uganda) using GBD 2023 data.

Two graph configurations are implemented:
- **Graph A** — Both-sex × Age × Country (75 observed nodes)
- **Graph B** — Male/Female × Age × Country (150 observed nodes)

Three model specifications are estimated for each configuration:
- SDI Only
- Risk Only (BMI, FPG, SBP)
- Full (SDI + Risk)

---

## Key Findings

| Disease | Best Improvement | Configuration | Interpretation |
|---------|-----------------|---------------|----------------|
| Stroke  | +15.9% [3.17, 27.72] | Graph A, Risk Only | Reliable spatial predictive gain |
| HHD     | −131.8% | All configurations | Absent spatial structure |
| IHD     | −0.9% to −77.1% | All configurations | No reliable spatial signal |
| Diabetes| −167% to −202% | All configurations | No spatial signal |

---

## Repository Structure

| File | Description | Run Order |
|------|-------------|-----------|
| `01_setup.py` | Dependencies, hyperparameters, constants | 1 |
| `02_data_loading.py` | Data loading, preprocessing, graph A/B construction | 2 |
| `03_graph_summary.py` | Graph structure summary printer | 3 |
| `04_graph_builders_model_utilities.py` | Graph builders, HGT model, training utilities | 4 |
| `05_degradation_diagnostic.py` | HHD & stroke degradation diagnostic | 5 |
| `06_ablation_study.py` | Ablation study (HHD & stroke) | 6 |
| `07_rolling_temporal_windows.py` | Rolling temporal window validation | 7 |
| `08_attention_visualisation.py` | Attention weight visualisation | 8 |
| `09_monte_carlo_uncertainty.py` | GBD Monte Carlo uncertainty propagation | 9 |
| `10_main_model_runner.py` | Main model runner — all models × both graphs | 10 |
| `11_combined_summary_export.py` | Combined summary, cross-graph comparison, export | 11 |

---

## Data Requirements

Data must be sourced from [IHME GBD Results Tool](https://vizhub.healthdata.org/gbd-results/):

| File | Description |
|------|-------------|
| `GBD_DataAll.csv` | Cause-specific mortality by country, age, sex, year |
| `SDI_GBD.csv` | Socio-demographic Index by country and year |

Place both files in your working directory before running.

---

## Requirements

```bash
pip install torch torch_geometric pysal libpysal esda spreg scikit-learn pandas numpy matplotlib seaborn
```

---

## Usage

Run files sequentially (01 through 11) in Google Colab or Jupyter. Each file depends on variables defined in previous files. The full pipeline can also be run via the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SallySims/HGT_GBD-Mortality/blob/main/GBDTransformers_FullRevised.ipynb)

---

## Citation

If you use this code, please cite:

> Simmons, S. (2026). Heterogeneous Graph Transformer Modelling of Spatial Patterns in NCD Mortality Across East Africa. *[Journal Name]*.

---

## License

MIT License. See LICENSE for details.
