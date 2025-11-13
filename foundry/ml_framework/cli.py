"""Command-line interface for ML framework.

Provides CLI commands for training models and managing experiments.
"""

import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_framework.data.loaders import CSVLoader
from ml_framework.data.transformers import StandardScalerTransformer, OneHotEncoderTransformer
from ml_framework.models import AutoModel, ModelTrainer, TrainingArguments
from ml_framework.artifacts import ArtifactStore


@click.group()
def cli() -> None:
    """ML Framework CLI - Train and manage ML models."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to training configuration file (YAML or JSON)",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("assignment"),
    help="Directory containing training data files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./experiments"),
    help="Directory to save experiment outputs",
)
@click.option(
    "--model-type",
    type=click.Choice(["logistic_regression", "random_forest"], case_sensitive=False),
    default="logistic_regression",
    help="Type of model to train",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing",
)
def train(
    config: Optional[Path],
    data_dir: Path,
    output_dir: Path,
    model_type: str,
    test_size: float,
) -> None:
    """Train a model using the ML framework.
    
    Examples:
        # Basic training
        ml-framework train --data-dir assignment --output-dir ./experiments
        
        # With custom model type
        ml-framework train --model-type random_forest --test-size 0.3
        
        # With config file
        ml-framework train --config config.yaml
    """
    click.echo("=" * 80)
    click.echo("ML Framework - Training")
    click.echo("=" * 80)
    
    # Load configuration if provided
    if config:
        click.echo(f"\nLoading configuration from {config}...")
        if config.suffix in [".yaml", ".yml"]:
            import yaml
            with open(config) as f:
                config_dict = yaml.safe_load(f)
        elif config.suffix == ".json":
            with open(config) as f:
                config_dict = json.load(f)
        else:
            raise click.BadParameter(f"Unsupported config format: {config.suffix}")
    else:
        config_dict = {}
    
    # Use config values or CLI defaults
    data_dir = Path(config_dict.get("data_dir", data_dir))
    output_dir = Path(config_dict.get("output_dir", output_dir))
    model_type = config_dict.get("model_type", model_type)
    test_size = config_dict.get("test_size", test_size)
    
    try:
        # Load data
        click.echo(f"\n1. Loading data from {data_dir}...")
        customers_path = data_dir / "customers_(4).csv"
        noncustomers_path = data_dir / "noncustomers_(4).csv"
        usage_path = data_dir / "usage_actions_(4).csv"
        
        if not all(p.exists() for p in [customers_path, noncustomers_path, usage_path]):
            raise FileNotFoundError("Required data files not found in data directory")
        
        # Prepare data (inline implementation)
        customers_loader = CSVLoader(customers_path)
        noncustomers_loader = CSVLoader(noncustomers_path)
        usage_loader = CSVLoader(usage_path)
        
        customers_df = customers_loader.load()
        noncustomers_df = noncustomers_loader.load()
        usage_df = usage_loader.load()
        
        # Add conversion target
        customers_df["conversion"] = 1
        noncustomers_df["conversion"] = 0
        
        # Combine customers and noncustomers
        all_companies = pd.concat([customers_df, noncustomers_df], ignore_index=True)
        
        # Aggregate usage actions by company ID
        usage_agg = usage_df.groupby("id").agg({
            "ACTIONS_CRM_CONTACTS": ["sum", "mean"],
            "ACTIONS_CRM_COMPANIES": ["sum", "mean"],
            "ACTIONS_CRM_DEALS": ["sum", "mean"],
            "ACTIONS_EMAIL": ["sum", "mean"],
            "USERS_CRM_CONTACTS": ["sum", "mean"],
            "USERS_CRM_COMPANIES": ["sum", "mean"],
            "USERS_CRM_DEALS": ["sum", "mean"],
            "USERS_EMAIL": ["sum", "mean"],
        }).reset_index()
        
        # Flatten column names
        usage_agg.columns = ["id"] + [
            f"{col[0]}_{col[1]}" for col in usage_agg.columns[1:]
        ]
        
        # Merge with company data
        merged_df = all_companies.merge(usage_agg, on="id", how="left")
        
        # Fill NaN values (companies with no usage data)
        usage_cols = [col for col in merged_df.columns if col.startswith(("ACTIONS_", "USERS_"))]
        merged_df[usage_cols] = merged_df[usage_cols].fillna(0)
        
        # Extract target
        y = merged_df["conversion"]
        X = merged_df.drop(columns=["conversion", "CLOSEDATE", "id"])
        
        # Drop MRR (only available for customers, would leak information)
        if "MRR" in X.columns:
            X = X.drop(columns=["MRR"])
        click.echo(f"   Loaded {len(X)} samples with {len(X.columns)} features")
        
        # Split data
        click.echo(f"\n2. Splitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        click.echo(f"   Training set: {len(X_train)} samples")
        click.echo(f"   Test set: {len(X_test)} samples")
        
        # Transform data
        click.echo("\n3. Transforming data...")
        import numpy as np
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScalerTransformer(columns=numeric_cols)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            encoder = OneHotEncoderTransformer(columns=categorical_cols)
            X_train_encoded = encoder.fit_transform(X_train_scaled)
            X_test_encoded = encoder.transform(X_test_scaled)
        else:
            X_train_encoded = X_train_scaled
            X_test_encoded = X_test_scaled
        
        X_train_encoded = X_train_encoded.fillna(0)
        X_test_encoded = X_test_encoded.fillna(0)
        
        click.echo(f"   Final feature count: {len(X_train_encoded.columns)}")
        
        # Train model
        click.echo(f"\n4. Training {model_type} model...")
        model = AutoModel.from_type(model_type, random_state=42, max_iter=1000)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir / "training"),
            num_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )
        
        trainer = ModelTrainer(
            model=model,
            args=training_args,
            train_data=(X_train_encoded, y_train),
            eval_data=(X_test_encoded, y_test),
        )
        
        result = trainer.train()
        click.echo(f"   Model trained successfully! Version: {result.version}")
        
        # Display metrics
        click.echo("\n5. Evaluation metrics:")
        for metric_name, metric_value in result.metrics.items():
            click.echo(f"   {metric_name}: {metric_value:.4f}")
        
        # Save artifacts
        click.echo("\n6. Saving artifacts...")
        store = ArtifactStore(base_dir=str(output_dir / "artifacts"))
        
        model_path = store.save_model(result.model, "cli_training", result.version)
        click.echo(f"   Model saved to: {model_path}")
        
        metrics_path = store.save_metrics(result.metrics, "cli_training", result.version)
        click.echo(f"   Metrics saved to: {metrics_path}")
        
        click.echo("\n" + "=" * 80)
        click.echo("Training completed successfully!")
        click.echo("=" * 80)
        
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--experiment-name",
    type=str,
    required=True,
    help="Name of the experiment",
)
@click.option(
    "--version",
    type=str,
    help="Version of the experiment (default: latest)",
)
def list_artifacts(experiment_name: str, version: Optional[str]) -> None:
    """List artifacts for an experiment.
    
    Examples:
        ml-framework list-artifacts --experiment-name conversion_prediction
        ml-framework list-artifacts --experiment-name conversion_prediction --version v1.0.0
    """
    store = ArtifactStore()
    
    if version:
        click.echo(f"Artifacts for {experiment_name} version {version}:")
        # List specific version artifacts
        experiment_dir = Path(store.base_dir) / experiment_name / version
        if experiment_dir.exists():
            for artifact_file in experiment_dir.rglob("*"):
                if artifact_file.is_file():
                    click.echo(f"  - {artifact_file.relative_to(store.base_dir)}")
        else:
            click.echo(f"No artifacts found for version {version}")
    else:
        click.echo(f"Available versions for {experiment_name}:")
        versions = store.list_versions(experiment_name)
        if versions:
            for v in versions:
                click.echo(f"  - {v}")
        else:
            click.echo(f"No versions found for {experiment_name}")


if __name__ == "__main__":
    cli()

