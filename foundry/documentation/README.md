# ML Framework for Data Scientists

A production-grade framework that enables data scientists to efficiently:
1. **Gather and transform data**
2. **Rapidly build and iterate on ML models**
3. **Host and deploy models**
4. **Measure model impact**

**Design Philosophy**: The framework emphasizes modularity, type safety, extensibility, and production-grade reliability.

---

##  Quick Start

### Installation

**Option 1: Using UV (Recommended)**

```bash
# Install UV if you haven't already
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies
uv sync

# Activate virtual environment
uv shell
```

**Option 2: Using Poetry**

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

**Option 3: Using pip and requirements.txt**

```bash
# Create virtual environment (Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package (editable mode for development)
pip install -e .

# Or install production dependencies only
pip install -r requirements.txt

# Or install with development dependencies
pip install -r requirements-dev.txt
```

**Option 4: Using Docker (Recommended for consistency)**

**From WSL (requires WSL 2 + Docker Desktop WSL integration):**
```bash
# Build and run the example
docker compose up --build

# Or run a specific example
docker compose run --rm ml-framework python examples/conversion_prediction.py
```

**Note**: If Docker is not available in WSL, use Docker Desktop from Windows PowerShell/CMD instead.

**Option 5: Testing Installation from GitHub**

To test that the package can be installed from GitHub:

```bash
# Using Docker Compose
docker compose -f docker-compose.git.yml up --build

```


### Basic Usage

**Option 1: Using Python API**

```python
from ml_framework.data.loaders import CSVLoader
from ml_framework.data.pipeline import DataPipeline
from ml_framework.data.transformers import StandardScalerTransformer
from ml_framework.models import AutoModel, ModelTrainer, TrainingArguments

# Load data
loader = CSVLoader("assignment/customers_(4).csv")
data = loader.load()

# Or use a pipeline
pipeline = DataPipeline([
    CSVLoader("assignment/customers_(4).csv"),
    StandardScalerTransformer(columns=["ALEXA_RANK"])
])
data = pipeline.execute()

# Train model
model = AutoModel.from_type("logistic_regression")
training_args = TrainingArguments(output_dir="./experiments")
trainer = ModelTrainer(model=model, args=training_args, train_data=(X_train, y_train))
result = trainer.train()
```

**Option 2: Using CLI**

```bash
# Train model via CLI (default: logistic_regression)
# Using UV (recommended):
uv run ml-framework train --data-dir assignment --output-dir ./experiments

# Or using Poetry:
poetry run ml-framework train --data-dir assignment --output-dir ./experiments

# With custom options
# Using UV (recommended):
uv run ml-framework train --model-type random_forest --test-size 0.3

# Or using Poetry:
poetry run ml-framework train --model-type random_forest --test-size 0.3

# With config file
# Using UV (recommended):
uv run ml-framework train --config config.yaml

# Or using Poetry:
poetry run ml-framework train --config config.yaml

# List artifacts
# Using UV (recommended):
uv run ml-framework list-artifacts --experiment-name conversion_prediction

# Or using Poetry:
poetry run ml-framework list-artifacts --experiment-name conversion_prediction
```

---

## Project Structure

```
ml-framework/
├── ml_framework/          # Main framework package
│   ├── data/              # Data pipeline module
│   │   ├── loaders/       # Data loaders (CSV, SQL, API)
│   │   └── transformers/  # Data transformers
│   ├── models/            # Model development module
│   │   ├── base_model.py  # Base model ABC
│   │   ├── auto.py        # AutoModel factory
│   │   ├── trainer.py     # Trainer class
│   │   └── training_arguments.py
│   ├── artifacts/         # Artifact management
│   ├── config/            # Configuration management
│   ├── schemas/           # Pydantic models
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
├── examples/              # Usage examples
├── pyproject.toml         # Project configuration (UV/Poetry)
└── README.md
```

---

##  Features

- CSV data loading
- Composable data pipelines
- Basic data transformations
- LogisticRegression model training
- Artifact saving (models, metrics, configs)
- CLI support for training
- Deployment and impact measurement interfaces (extensible)

---

##  Documentation

- **[QUICK_START.md](./QUICK_START.md)**: Quick start guide with installation and usage examples
- **[ADDING_CUSTOM_MODELS.md](/ADDING_CUSTOM_MODELS.md)**: Guide for extending the framework with custom models

---


