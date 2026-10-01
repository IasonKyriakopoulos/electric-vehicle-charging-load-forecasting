# Electric Vehicle Charging Load Forecasting: An Experimental Comparison of Machine Learning Methods

This repository accompanies the paper:

**I. Kyriakopoulos and Y. Theodoridis, "Electric Vehicle Charging Load Forecasting: An Experimental Comparison of Machine Learning Methods" (2026).**

**Paper:** [arXiv:2512.17257](https://arxiv.org/abs/2512.17257)

It provides the code, configuration, and instructions required to reproduce the experimental pipeline and regenerate the processed datasets used in the study.

---

## Repository Structure

- **`data/`**  
  Contains folder structure for **raw** and **processed** datasets for each city used in the study.

- **`scripts/`**  
  The preprocessing, forecasting, and metrics Python scripts:
  - **Preprocessing**: data cleaning and transformation for each city
  - **Forecasting**: model training and forecasting
  - **Metrics**: error calculation and summary of models

- **`results/`**  
  Stores model prediction outputs and evaluation metrics.

- **`envs/`**
  Contains environment files.

- **`run_all.sh`**
  One-command pipeline runner.

---

## Datasets

The **publicly available** EV-charging datasets used in this research:  
[https://github.com/yvenn-amara/ev-load-open-data](https://github.com/yvenn-amara/ev-load-open-data)

**Important:**  
Due to file-size limits, large processed “.pkl” files are **NOT** tracked on GitHub.
To regenerate them, run the preprocessing step (see below) using the provided raw data paths.

---

## Requirements

All experiments were executed under Python 3.8.10 inside a Singularity container (tf instance).

Main libraries:

- tensorflow==2.13.0
- xgboost==2.1.4
- scikit-learn==1.3.2
- statsmodels==0.14.1
- pmdarima==2.0.4
- pandas==2.0.3
- numpy==1.24.3
- holidays==0.58

Exact dependency snapshot: envs/requirements.txt

---

## How to Reproduce the Pipeline

```bash
# 1. Create and activate a Python environment (Python 3.8 or later)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install all required dependencies
pip install -r envs/requirements.txt

# 3. Run the complete pipeline (preprocessing → training → metrics)
bash run_all.sh
```

This script automatically:
- Preprocesses all city datasets
- Trains forecasting models
- Computes metrics and saves results under the "results/" directory

Note:
The original experiments were executed in a Singularity container on a university GPU cluster to ensure consistent CUDA and TensorFlow versions.
The commands above reproduce the experimental workflow on a compatible Python environment. Exact numerical results may vary slightly across hardware, TensorFlow, and CUDA configurations.

---

## Citation

If you use this code or reproduce these results, please cite:

I. Kyriakopoulos, Y. Theodoridis,
"Electric Vehicle Charging Load Forecasting: An Experimental Comparison of Machine Learning Methods", 2026.