# ML Framework — Quick Start Guide

##  Getting Started

### 1. Install Dependencies

**Option 1: Using UV (Recommended)**

```bash
# Install UV if you haven't already
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Navigate to project directory (if needed)
# cd foundry

# Install dependencies
uv sync

# Activate virtual environment
uv shell
```

**Option 2: Using Poetry**

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Navigate to project directory (if needed)
# cd foundry

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

**Option 3: Using pip and requirements.txt**

```bash
# Navigate to project directory (if needed)
# cd foundry

# Create virtual environment (Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -r requirements-dev.txt
```

### 2. Run the Example

**Option 1: Using Python Script**

```bash
# Run the end-to-end example
# Using UV (recommended):
uv run python examples/conversion_prediction.py

# Or using Poetry:
poetry run python examples/conversion_prediction.py
```

**Option 2: Using CLI**

```bash
# Train model via CLI (default: logistic_regression)
# Using UV (recommended):
uv run ml-framework train --data-dir assignment --output-dir ./experiments

# Or using Poetry:
poetry run ml-framework train --data-dir assignment --output-dir ./experiments

# Train with a different model type (e.g., random_forest)
# Using UV (recommended):
uv run ml-framework train --data-dir assignment --output-dir ./experiments --model-type random_forest

# Or using Poetry:
poetry run ml-framework train --data-dir assignment --output-dir ./experiments --model-type random_forest

# List artifacts
# Using UV (recommended):
uv run ml-framework list-artifacts --experiment-name conversion_prediction

# Or using Poetry:
poetry run ml-framework list-artifacts --experiment-name conversion_prediction
```

This will (by default, trains a **logistic regression** model):
- Load CSV data (customers, noncustomers, usage_actions)
- Transform and prepare data
- Train a LogisticRegression model
- Evaluate the model
- Save artifacts (model, metrics, configs)

### 3. Run Tests

```bash
# Run all tests
# Using UV (recommended):
uv run pytest

# Or using Poetry:
poetry run pytest

# Run with coverage
# Using UV (recommended):
uv run pytest --cov=ml_framework --cov-report=html

# Or using Poetry:
poetry run pytest --cov=ml_framework --cov-report=html
```

---

##  Basic Usage

### Load Data

```python
from ml_framework.data.loaders import CSVLoader

# Load CSV file
loader = CSVLoader("assignment/customers_(4).csv")
data = loader.load()
print(data.head())
```

### Transform Data

```python
from ml_framework.data.transformers import StandardScalerTransformer, OneHotEncoderTransformer

# Scale numeric features
scaler = StandardScalerTransformer(columns=["ALEXA_RANK"])
X_scaled = scaler.fit_transform(X_train)

# Encode categorical features
encoder = OneHotEncoderTransformer(columns=["INDUSTRY"])
X_encoded = encoder.fit_transform(X_scaled)
```

### Train Model

```python
from ml_framework.models import AutoModel, ModelTrainer, TrainingArguments

# Create model
model = AutoModel.from_type("logistic_regression")

# Configure training
training_args = TrainingArguments(
    output_dir="./experiments/conversion_model",
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# Train model
trainer = ModelTrainer(
    model=model,
    args=training_args,
    train_data=(X_train, y_train),
    eval_data=(X_test, y_test)
)
result = trainer.train()

# View metrics
print(result.metrics)
```

### Save Artifacts

```python
from ml_framework.artifacts import ArtifactStore

# Save model and metrics
store = ArtifactStore()
store.save_model(model, "conversion_model", result.version)
store.save_metrics(result.metrics, "conversion_model", result.version)
```

---

## Development

### Code Quality

```bash
# Format code
# Using UV (recommended):
uv run black ml_framework tests examples

# Or using Poetry:
poetry run black ml_framework tests examples

# Lint code
# Using UV (recommended):
uv run ruff check ml_framework tests examples

# Or using Poetry:
poetry run ruff check ml_framework tests examples

# Type check
# Using UV (recommended):
uv run mypy ml_framework

# Or using Poetry:
poetry run mypy ml_framework
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

---

##  Next Steps

1. **Review the example**: `examples/conversion_prediction.py`
2. **Read the documentation**:
   - [ADDING_CUSTOM_MODELS.md](/ADDING_CUSTOM_MODELS.md)
3. **Extend the framework**: See extension examples in [ADDING_CUSTOM_MODELS.md](/ADDING_CUSTOM_MODELS.md)

---

## Questions?
- Check [ADDING_CUSTOM_MODELS.md](/ADDING_CUSTOM_MODELS.md) for extending the framework
- Check [examples/](./examples/) for usage examples

