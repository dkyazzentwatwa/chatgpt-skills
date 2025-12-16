#!/usr/bin/env python3
"""Model Comparison Tool - Compare multiple ML models systematically."""

import argparse
import json
import os
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


class ModelComparisonTool:
    """Compare multiple ML models."""

    CLASSIFIERS = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42, probability=True)
    }

    REGRESSORS = {
        'rf': RandomForestRegressor(random_state=42),
        'gb': GradientBoostingRegressor(random_state=42),
        'lr': LinearRegression(),
        'svr': SVR()
    }

    def __init__(self):
        self.X = None
        self.y = None
        self.task = None
        self.results = {}

    def load_data(self, X, y, task='classification'):
        """Load data and set task type."""
        self.X = X
        self.y = y
        self.task = task
        return self

    def compare_models(self, models: List[str] = None, cv_folds: int = 5) -> Dict:
        """Compare multiple models with cross-validation."""
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if models is None:
            models = list(self.CLASSIFIERS.keys() if self.task == 'classification' else self.REGRESSORS.keys())

        model_dict = self.CLASSIFIERS if self.task == 'classification' else self.REGRESSORS
        cv = StratifiedKFold(n_splits=cv_folds) if self.task == 'classification' else KFold(n_splits=cv_folds)

        scoring = ['accuracy', 'f1_weighted'] if self.task == 'classification' else ['neg_mean_squared_error', 'r2']

        print(f"Comparing {len(models)} models with {cv_folds}-fold cross-validation...\n")

        for model_name in models:
            if model_name not in model_dict:
                print(f"Warning: Unknown model '{model_name}', skipping")
                continue

            model = model_dict[model_name]
            print(f"Evaluating {model_name}...")

            cv_results = cross_validate(model, self.X, self.y, cv=cv, scoring=scoring, return_train_score=True)

            self.results[model_name] = {
                'train_score': cv_results['train_' + scoring[0].replace('neg_', '')].mean(),
                'test_score': cv_results['test_' + scoring[0].replace('neg_', '')].mean(),
                'test_std': cv_results['test_' + scoring[0].replace('neg_', '')].std(),
                'fit_time': cv_results['fit_time'].mean(),
                'score_time': cv_results['score_time'].mean()
            }

        return self.results

    def get_best_model(self, metric: str = 'test_score'):
        """Get best model based on metric."""
        if not self.results:
            raise ValueError("No results available. Run compare_models() first.")

        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0]

    def print_results(self):
        """Print comparison results."""
        if not self.results:
            print("No results to display")
            return

        df = pd.DataFrame(self.results).T
        print("\nModel Comparison Results:")
        print("=" * 80)
        print(df.round(4))
        print("\nBest model:", self.get_best_model())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare ML models')
    parser.add_argument('--data', required=True, help='CSV data file')
    parser.add_argument('--target', required=True, help='Target column')
    parser.add_argument('--task', choices=['classification', 'regression'], default='classification')
    parser.add_argument('--models', default='rf,gb,lr', help='Comma-separated model names')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    comparator = ModelComparisonTool()
    comparator.load_data(X, y, task=args.task)
    comparator.compare_models(models=args.models.split(','), cv_folds=args.cv)
    comparator.print_results()
