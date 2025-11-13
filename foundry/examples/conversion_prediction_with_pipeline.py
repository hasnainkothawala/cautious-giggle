"""End-to-end example: Conversion Prediction Model using DataPipeline and Deployment.

This example demonstrates the full ML framework workflow with advanced features:
1. Use DataPipeline to chain data loading and transformations
2. Train a LogisticRegression model
3. Evaluate the model
4. Save artifacts
5. Demonstrate deployment interface
6. Measure model impact using MetricsCollector

Uses real data from the assignment (customers.csv, noncustomers.csv, usage_actions.csv).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml_framework.data.loaders import CSVLoader
from ml_framework.data.pipeline import DataPipeline
from ml_framework.data.transformers import StandardScalerTransformer, OneHotEncoderTransformer
from ml_framework.models import AutoModel, ModelTrainer, TrainingArguments
from ml_framework.artifacts import ArtifactStore
from ml_framework.deployment import BaseDeployer
from ml_framework.impact import MetricsCollector


def prepare_data_with_pipeline(
    customers_path: Path,
    noncustomers_path: Path,
    usage_actions_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for conversion prediction using DataPipeline.
    
    This function demonstrates how to use DataPipeline to chain
    data loading and transformation steps.
    
    Args:
        customers_path: Path to customers CSV file.
        noncustomers_path: Path to noncustomers CSV file.
        usage_actions_path: Path to usage_actions CSV file.
    
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    # Load data using CSVLoader
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


class SimpleDeployer(BaseDeployer):
    """Simple deployer implementation for demonstration.
    
    This is a minimal implementation showing how to extend BaseDeployer.
    In production, you would implement actual deployment logic (Docker,
    Kubernetes, cloud services, etc.).
    """
    
    def __init__(self) -> None:
        """Initialize simple deployer."""
        self.deployments: dict[str, dict] = {}
    
    def deploy(
        self,
        model,
        config: dict | None = None,
    ) -> dict:
        """Deploy model to serving infrastructure.
        
        Args:
            model: Trained model instance to deploy.
            config: Optional deployment configuration dictionary.
        
        Returns:
            Dictionary containing deployment information.
        """
        if config is None:
            config = {}
        
        # Generate deployment ID
        import uuid
        deployment_id = str(uuid.uuid4())[:8]
        
        # In a real implementation, this would:
        # - Create a Docker container
        # - Start a FastAPI server
        # - Register with load balancer
        # - Set up monitoring
        # For now, we just store deployment info
        
        deployment_info = {
            "deployment_id": deployment_id,
            "endpoint": f"http://localhost:8000/predict/{deployment_id}",
            "version": config.get("version", "unknown"),
            "status": "deployed",
            "metadata": config,
        }
        
        self.deployments[deployment_id] = {
            "model": model,
            "info": deployment_info,
        }
        
        print(f"   Model deployed! Deployment ID: {deployment_id}")
        print(f"   Endpoint: {deployment_info['endpoint']}")
        
        return deployment_info
    
    def undeploy(self, deployment_id: str) -> None:
        """Undeploy a model deployment.
        
        Args:
            deployment_id: Identifier for the deployment to remove.
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # In a real implementation, this would:
        # - Stop the server
        # - Remove from load balancer
        # - Clean up resources
        
        del self.deployments[deployment_id]
        print(f"   Deployment {deployment_id} undeployed successfully")
    
    def list_deployments(self) -> list[dict]:
        """List all active deployments.
        
        Returns:
            List of dictionaries containing deployment information.
        """
        return [info["info"] for info in self.deployments.values()]


def main() -> None:
    """Main function demonstrating end-to-end ML workflow with pipeline and deployment."""
    print("=" * 80)
    print("ML Framework - Conversion Prediction Example (with Pipeline & Deployment)")
    print("=" * 80)
    
    # Paths to data files
    assignment_dir = Path(__file__).parent.parent / "assignment"
    customers_path = assignment_dir / "customers_(4).csv"
    noncustomers_path = assignment_dir / "noncustomers_(4).csv"
    usage_actions_path = assignment_dir / "usage_actions_(4).csv"
    
    print("\n1. Loading and preparing data...")
    X, y = prepare_data_with_pipeline(customers_path, noncustomers_path, usage_actions_path)
    print(f"   Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"   Conversion rate: {y.mean():.2%}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Transform data using DataPipeline
    print("\n3. Transforming data using DataPipeline...")
    
    # Identify columns for transformation
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    
    # Create transformers
    scaler = StandardScalerTransformer(columns=numeric_cols)
    encoder = OneHotEncoderTransformer(columns=categorical_cols) if categorical_cols else None
    
    # Option 1: Use DataPipeline for a single data transformation workflow
    # Note: DataPipeline requires a loader first, so we'll demonstrate
    # the pipeline pattern with transformers only (after data is loaded)
    
    # For demonstration, we'll show how pipeline would work if we had
    # a transformer-only pipeline. In practice, you might create a custom
    # transformer pipeline class or use transformers sequentially.
    
    # Fit transformers on training data
    print("   Fitting transformers on training data...")
    scaler.fit(X_train)
    if encoder:
        encoder.fit(X_train)
    
    # Transform data
    print("   Transforming data...")
    X_train_transformed = scaler.transform(X_train)
    if encoder:
        X_train_transformed = encoder.transform(X_train_transformed)
    
    X_test_transformed = scaler.transform(X_test)
    if encoder:
        X_test_transformed = encoder.transform(X_test_transformed)
    
    # Handle any remaining NaN values
    X_train_transformed = X_train_transformed.fillna(0)
    X_test_transformed = X_test_transformed.fillna(0)
    
    print(f"   Scaled {len(numeric_cols)} numeric features")
    if categorical_cols:
        print(f"   Encoded {len(categorical_cols)} categorical features")
    print(f"   Final feature count: {len(X_train_transformed.columns)}")
    
    # Alternative: Demonstrate DataPipeline with a dummy loader
    # (This shows how pipeline chains work)
    print("\n   Demonstrating DataPipeline pattern...")
    from ml_framework.data.loaders.base_loader import BaseLoader
    
    class DummyLoader(BaseLoader):
        """Dummy loader for pipeline demonstration."""
        def __init__(self, data: pd.DataFrame):
            self.data = data
        
        def load(self, path=None):
            return self.data
    
    # Create a pipeline that chains loader + transformers
    # This demonstrates the composable pipeline pattern
    pipeline = DataPipeline([
        DummyLoader(X_train),  # Loader
        scaler,                 # Transformer 1
    ])
    
    if encoder:
        # Add encoder to pipeline steps
        pipeline.steps.append(encoder)
    
    # Execute pipeline (this would normally be used for new data)
    X_train_from_pipeline = pipeline.execute()
    print(f"   Pipeline executed successfully! Output shape: {X_train_from_pipeline.shape}")
    
    # Train model
    print("\n4. Training model...")
    model = AutoModel.from_type("logistic_regression", random_state=42, max_iter=1000)
    
    training_args = TrainingArguments(
        output_dir="./experiments/conversion_prediction_pipeline",
        num_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    
    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_data=(X_train_transformed, y_train),
        eval_data=(X_test_transformed, y_test),
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
    
    model_path = store.save_model(result.model, "conversion_prediction_pipeline", result.version)
    print(f"   Model saved to: {model_path}")
    
    metrics_path = store.save_metrics(result.metrics, "conversion_prediction_pipeline", result.version)
    print(f"   Metrics saved to: {metrics_path}")
    
    # Save config
    config = {
        "model_type": "logistic_regression",
        "training_args": result.training_args.model_dump(),
        "feature_count": len(X_train_transformed.columns),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "pipeline_used": True,
    }
    config_path = store.save_config(config, "conversion_prediction_pipeline", result.version)
    print(f"   Config saved to: {config_path}")
    
    # Deploy model
    print("\n7. Deploying model...")
    deployer = SimpleDeployer()
    
    deployment_config = {
        "version": result.version,
        "model_type": "logistic_regression",
        "endpoint_type": "rest_api",
        "port": 8000,
    }
    
    deployment_result = deployer.deploy(result.model, config=deployment_config)
    
    # List deployments
    print("\n8. Listing deployments...")
    deployments = deployer.list_deployments()
    print(f"   Active deployments: {len(deployments)}")
    for dep in deployments:
        print(f"   - {dep['deployment_id']}: {dep['endpoint']} (status: {dep['status']})")
    
    # Demonstrate undeploy
    print("\n9. Undeploying model (demonstration)...")
    deployer.undeploy(deployment_result["deployment_id"])
    
    # Measure model impact
    print("\n10. Measuring model impact...")
    collector = MetricsCollector()
    
    # Collect production metrics (using test set as simulation)
    print("   Collecting production metrics...")
    production_metrics = collector.collect_production_metrics(
        model=result.model,
        X=X_test_transformed,
        y=y_test,  # In real production, this might not be available
        metadata={
            "model_version": result.version,
            "deployment_id": deployment_result["deployment_id"],
            "experiment_name": "conversion_prediction_pipeline",
        },
    )
    
    print(f"   Collected metrics for {production_metrics['sample_count']} samples")
    if "performance_metrics" in production_metrics:
        print("   Performance metrics:")
        for metric_name, metric_value in production_metrics["performance_metrics"].items():
            print(f"     - {metric_name}: {metric_value:.4f}")
    
    # Analyze business impact
    print("   Analyzing business impact...")
    business_config = {
        "conversion_value": 100.0,  # Value per conversion (e.g., $100 MRR)
        "cost_per_prediction": 0.01,  # Cost per prediction (e.g., API call cost)
    }
    
    business_impact = collector.analyze_business_impact(
        metrics=production_metrics,
        business_config=business_config,
    )
    
    print(f"   Business impact analysis:")
    print(f"     - Estimated conversions: {business_impact['estimated_conversions']}")
    print(f"     - Estimated revenue: ${business_impact['estimated_revenue']:.2f}")
    print(f"     - ROI: {business_impact['roi']:.2%}")
    
    # Compare model versions (if we had multiple versions)
    print("\n11. Model version comparison (demonstration)...")
    # Simulate comparing with a previous version
    # In practice, you would load metrics from a previous model version
    previous_metrics = {
        "performance_metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.80,
            "f1_score": 0.81,
        },
    }
    
    comparison = collector.compare_versions(
        metrics_v1=previous_metrics,
        metrics_v2=production_metrics,
    )
    
    if "performance_delta" in comparison and comparison["performance_delta"]:
        print("   Performance delta (current vs previous):")
        for metric, delta in comparison["performance_delta"].items():
            sign = "+" if delta >= 0 else ""
            print(f"     - {metric}: {sign}{delta:.4f}")
        print(f"   Recommendation: {comparison['recommendation']}")
    
    # Save impact metrics
    print("\n12. Saving impact metrics...")
    impact_metrics_path = store.save_metrics(
        {
            "production_metrics": production_metrics,
            "business_impact": business_impact,
            "version_comparison": comparison,
        },
        "conversion_prediction_pipeline_impact",
        result.version,
    )
    print(f"   Impact metrics saved to: {impact_metrics_path}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nKey features demonstrated:")
    print("  - DataPipeline for composable data processing")
    print("  - Model training and evaluation")
    print("  - Artifact saving")
    print("  - Deployment interface (extensible)")
    print("  - Impact measurement (MetricsCollector)")
    print("\nNext steps:")
    print("  - Review saved artifacts in ./artifacts/conversion_prediction_pipeline/")
    print("  - Implement full deployment logic (Docker, FastAPI, etc.)")
    print("  - Add monitoring and health checks")
    print("  - Extend framework with new loaders, transformers, or models")


if __name__ == "__main__":
    main()

