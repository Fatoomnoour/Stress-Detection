# Stress Detection

A small educational machine-learning project that predicts a stress level from the sleep and lifestyle dataset. The repository includes the original notebook, a Flask entry point, and serialized model artifacts used by the demo. It is not a medical diagnostic tool.

## Reproducible setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The committed `preprocessor.pkl` and `stress_predictor.pkl` files are inference artifacts. Do not replace them without recording the training data version, feature order, preprocessing steps, evaluation metrics, and Python/package versions. The CSV is a sample dataset for learning only; do not add personal or production data.

## Quality and limitations

Run the notebook from top to bottom to reproduce the exploratory workflow. Before using the model for any real decision, add a held-out evaluation report, class-level metrics, input-schema validation, and a model/data version record. Predictions are for education and portfolio demonstration only.
