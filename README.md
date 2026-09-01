# Diabetes 30-Day Readmission ML System

An end-to-end machine-learning project for predicting whether a diabetes-related
hospital encounter will result in readmission within 30 days, using the UCI
Diabetes 130-US Hospitals dataset.

## Layout

```text
.
|-- configs/             # Reproducible experiment settings
|-- data/
|   |-- raw/             # Original UCI files (never modified)
|   |-- interim/         # Validated and cleaned data
|   `-- processed/       # Model-ready train/test datasets
|-- models/              # Serialized pipelines and model artifacts
|-- notebooks/           # Exploration only; production logic belongs in src
|-- reports/
|   `-- figures/         # Metrics, plots, and model reports
|-- src/
|   |-- config.py        # Paths and shared constants
|   |-- ingestion.py     # Dataset acquisition/loading
|   |-- validation.py    # Schema and quality checks
|   |-- preprocessing.py # Cleaning and target creation
|   |-- features.py      # Feature transformation pipeline
|   |-- train.py         # Training and model persistence
|   |-- evaluate.py      # Evaluation metrics
|   `-- predict.py       # Batch inference
|-- tests/               # Automated tests
`-- pyproject.toml       # Package metadata and dependencies
```

## Getting started

1. Download the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from UCI.
2. Put `diabetic_data.csv` in `data/raw/`.
3. Create an environment and install the project with `pip install -e ".[dev]"`.
4. Run the tests with `pytest`.

The initial model is a logistic-regression baseline. All preprocessing is kept
inside the fitted scikit-learn pipeline to reduce training/serving skew and data leakage.
