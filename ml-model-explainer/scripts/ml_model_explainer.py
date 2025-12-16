#!/usr/bin/env python3
"""
ML Model Explainer - Explain model predictions using SHAP.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


class MLModelExplainer:
    """Explain ML model predictions."""

    def __init__(self):
        """Initialize explainer."""
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_background = None

    def load_model(self, model: Any, X_background: pd.DataFrame = None) -> 'MLModelExplainer':
        """Load model and create explainer."""
        self.model = model
        self.X_background = X_background

        # Create appropriate explainer
        if hasattr(model, 'tree_'):
            # Tree-based model
            self.explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'coef_'):
            # Linear model
            self.explainer = shap.LinearExplainer(model, X_background)
        else:
            # General explainer (slower)
            self.explainer = shap.KernelExplainer(model.predict, X_background)

        return self

    def explain(self, X: pd.DataFrame, check_additivity: bool = False) -> np.ndarray:
        """Calculate SHAP values."""
        self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        return self.shap_values

    def plot_waterfall(self, instance_index: int, output: str) -> str:
        """Plot waterfall chart for single prediction."""
        if self.shap_values is None:
            raise ValueError("Call explain() first")

        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_index],
                base_values=self.explainer.expected_value,
                data=self.X_background.iloc[instance_index] if self.X_background is not None else None
            ),
            show=False
        )

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def plot_summary(self, output: str) -> str:
        """Plot summary of all SHAP values."""
        if self.shap_values is None:
            raise ValueError("Call explain() first")

        shap.summary_plot(self.shap_values, self.X_background, show=False)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def feature_importance(self) -> pd.DataFrame:
        """Get global feature importance."""
        if self.shap_values is None:
            raise ValueError("Call explain() first")

        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame({
            'feature': self.X_background.columns if self.X_background is not None else range(len(importance)),
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df


def main():
    parser = argparse.ArgumentParser(description="ML Model Explainer")

    parser.add_argument("--model", required=True, help="Pickled model file")
    parser.add_argument("--data", required=True, help="Data CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")

    args = parser.parse_args()

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Load data
    X = pd.read_csv(args.data)

    # Create explainer
    explainer = MLModelExplainer()
    explainer.load_model(model, X)

    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer.explain(X)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save plots
    print("Generating visualizations...")
    explainer.plot_summary(str(output_path / 'summary.png'))
    print(f"Summary plot saved: {output_path / 'summary.png'}")

    # Feature importance
    importance = explainer.feature_importance()
    importance.to_csv(output_path / 'feature_importance.csv', index=False)
    print(f"Feature importance saved: {output_path / 'feature_importance.csv'}")

    print(f"\nTop 5 important features:")
    print(importance.head())


if __name__ == "__main__":
    main()
