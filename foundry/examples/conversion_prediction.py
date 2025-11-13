"""End-to-end example: Conversion Prediction Model.

This example demonstrates the full ML framework workflow:
1. Load data from CSV files
2. Transform and prepare data
3. Train a LogisticRegression model
4. Evaluate the model
5. Save artifacts

Uses real data from the assignment (customers.csv, noncustomers.csv, usage_actions.csv).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml_framework.data.loaders import CSVLoader
from ml_framework.data.transformers import StandardScalerTransformer, OneHotEncoderTransformer
from ml_framework.models import AutoModel, ModelTrainer, TrainingArguments
from ml_framework.artifacts import ArtifactStore


def prepare_data(
    customers_path: Path,
    noncustomers_path: Path,
    usage_actions_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for conversion prediction.
    
    Args:
        customers_path: Path to customers CSV file.
        noncustomers_path: Path to noncustomers CSV file.
        usage_actions_path: Path to usage_actions CSV file.
    
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    # Load data
    customers_loader = CSVLoader(customers_path)
    noncustomers_loader = CSVLoader(noncustomers_path)
    usage_loader = CSVLoader(usage_actions_path)
    
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
    
    return X, y


def main() -> None:
    """Main function demonstrating end-to-end ML workflow."""
    print("=" * 80)
    print("ML Framework - Conversion Prediction Example")
    print("=" * 80)
    
    # Paths to data files
    assignment_dir = Path(__file__).parent.parent / "assignment"
    customers_path = assignment_dir / "customers_(4).csv"
    noncustomers_path = assignment_dir / "noncustomers_(4).csv"
    usage_actions_path = assignment_dir / "usage_actions_(4).csv"
    
    print("\n1. Loading and preparing data...")
    X, y = prepare_data(customers_path, noncustomers_path, usage_actions_path)
    print(f"   Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"   Conversion rate: {y.mean():.2%}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Transform data
    print("\n3. Transforming data...")
    
    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScalerTransformer(columns=numeric_cols)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode categorical features
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        encoder = OneHotEncoderTransformer(columns=categorical_cols)
        X_train_encoded = encoder.fit_transform(X_train_scaled)
        X_test_encoded = encoder.transform(X_test_scaled)
    else:
        X_train_encoded = X_train_scaled
        X_test_encoded = X_test_scaled
    
    # Handle any remaining NaN values (fill with 0)
    X_train_encoded = X_train_encoded.fillna(0)
    X_test_encoded = X_test_encoded.fillna(0)
    
    print(f"   Scaled {len(numeric_cols)} numeric features")
    if categorical_cols:
        print(f"   Encoded {len(categorical_cols)} categorical features")
    print(f"   Final feature count: {len(X_train_encoded.columns)}")
    
    # Train model
    print("\n4. Training model...")
    model = AutoModel.from_type("logistic_regression", random_state=42, max_iter=1000)
    
    training_args = TrainingArguments(
        output_dir="./experiments/conversion_prediction",
        num_epochs=1,  # LogisticRegression doesn't use epochs, but kept for consistency
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
    print(f"   Model trained successfully!")
    print(f"   Version: {result.version}")
    
    # Display metrics
    print("\n5. Evaluation metrics:")
    for metric_name, metric_value in result.metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    # Save artifacts
    print("\n6. Saving artifacts...")
    store = ArtifactStore(base_dir="./artifacts")
    
    model_path = store.save_model(result.model, "conversion_prediction", result.version)
    print(f"   Model saved to: {model_path}")
    
    metrics_path = store.save_metrics(result.metrics, "conversion_prediction", result.version)
    print(f"   Metrics saved to: {metrics_path}")
    
    # Save config
    config = {
        "model_type": "logistic_regression",
        "training_args": result.training_args.model_dump(),
        "feature_count": len(X_train_encoded.columns),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }
    config_path = store.save_config(config, "conversion_prediction", result.version)
    print(f"   Config saved to: {config_path}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)



if __name__ == "__main__":
    main()

