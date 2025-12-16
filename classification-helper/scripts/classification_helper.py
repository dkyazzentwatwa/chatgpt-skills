#!/usr/bin/env python3
"""
Classification Helper - Quick classifier with evaluation.
"""

import argparse
import pickle
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationHelper:
    """Helper for classification tasks."""

    MODELS = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42, probability=True)
    }

    def __init__(self):
        """Initialize helper."""
        self.model = None
        self.model_name = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, X: pd.DataFrame, y: pd.Series,
                  test_size: float = 0.2) -> 'ClassificationHelper':
        """Load and split data."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return self

    def train(self, model_type: str = 'rf', **kwargs) -> 'ClassificationHelper':
        """Train model."""
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model: {model_type}")

        self.model = self.MODELS[model_type]
        if kwargs:
            self.model.set_params(**kwargs)

        self.model.fit(self.X_train, self.y_train)
        self.model_name = model_type

        return self

    def evaluate(self) -> Dict:
        """Evaluate model."""
        if self.model is None:
            raise ValueError("Train model first")

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)

        # Metrics
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # ROC-AUC (for binary or multiclass)
        try:
            if len(np.unique(self.y_train)) == 2:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = None

        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)

        return {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'report': report
        }

    def plot_confusion_matrix(self, output: str) -> str:
        """Plot confusion matrix."""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance (for tree-based models)."""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def save_model(self, output: str) -> str:
        """Save trained model."""
        with open(output, 'wb') as f:
            pickle.dump(self.model, f)
        return output


def main():
    parser = argparse.ArgumentParser(description="Classification Helper")

    parser.add_argument("--data", required=True, help="Training data CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--model", default='rf',
                       choices=['rf', 'gb', 'lr', 'svm'],
                       help="Model type")
    parser.add_argument("--output", "-o", required=True, help="Output model file")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Classes: {y.value_counts().to_dict()}")

    # Train
    helper = ClassificationHelper()
    helper.load_data(X, y)

    print(f"\nTraining {args.model} model...")
    helper.train(model_type=args.model)

    # Evaluate
    results = helper.evaluate()

    print(f"\nModel Performance:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1-Score: {results['f1']:.3f}")
    if results['roc_auc']:
        print(f"  ROC-AUC: {results['roc_auc']:.3f}")
    print(f"  CV Score: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")

    # Save model
    helper.save_model(args.output)
    print(f"\nModel saved: {args.output}")

    # Feature importance
    importance = helper.feature_importance()
    if importance is not None:
        print(f"\nTop 5 features:")
        print(importance.head())


if __name__ == "__main__":
    main()
