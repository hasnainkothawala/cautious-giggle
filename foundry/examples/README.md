# ML Framework Examples

This directory contains example scripts demonstrating how to use the ML Framework.

## Examples

### `conversion_prediction.py`

End-to-end example demonstrating:
- Loading CSV data (customers, noncustomers, usage_actions)
- Data transformation (scaling, encoding)
- Training a LogisticRegression model
- Model evaluation
- Artifact saving

**Usage**:
```bash
cd infer-to-prod
poetry run python examples/conversion_prediction.py
```

**What it does**:
1. Loads the three CSV files from the assignment
2. Merges and prepares data for conversion prediction
3. Trains a LogisticRegression model
4. Evaluates the model on test data
5. Saves model, metrics, and config artifacts

---

## Running Examples

Make sure you have:
1. Installed dependencies: `poetry install`
2. Activated virtual environment: `poetry shell`
3. Data files in `assignment/` directory

Then run:
```bash
poetry run python examples/conversion_prediction.py
```

