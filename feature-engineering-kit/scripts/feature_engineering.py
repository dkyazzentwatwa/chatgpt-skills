#!/usr/bin/env python3
"""
Feature Engineering Kit - Auto-generate features.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


class FeatureEngineeringKit:
    """Feature engineering utilities."""

    def __init__(self):
        """Initialize kit."""
        self.df = None
        self.transformers = {}

    def load_data(self, df: pd.DataFrame) -> 'FeatureEngineeringKit':
        """Load data."""
        self.df = df.copy()
        return self

    def encode_categorical(self, columns: list = None, method: str = 'onehot') -> 'FeatureEngineeringKit':
        """Encode categorical variables."""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns

        for col in columns:
            if method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
            elif method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.transformers[f'{col}_encoder'] = le

        return self

    def scale_features(self, columns: list = None, method: str = 'standard') -> 'FeatureEngineeringKit':
        """Scale numeric features."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.transformers['scaler'] = scaler

        return self

    def handle_missing(self, strategy: str = 'mean') -> 'FeatureEngineeringKit':
        """Handle missing values."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        return self

    def create_polynomial(self, columns: list, degree: int = 2) -> 'FeatureEngineeringKit':
        """Create polynomial features."""
        for col in columns:
            for d in range(2, degree + 1):
                self.df[f'{col}_power{d}'] = self.df[col] ** d

        return self

    def create_interactions(self, column_pairs: list) -> 'FeatureEngineeringKit':
        """Create interaction features."""
        for col1, col2 in column_pairs:
            self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]

        return self

    def bin_feature(self, column: str, bins: int = 5, labels: list = None) -> 'FeatureEngineeringKit':
        """Bin continuous feature."""
        self.df[f'{column}_binned'] = pd.cut(self.df[column], bins=bins, labels=labels)

        return self

    def get_data(self) -> pd.DataFrame:
        """Get transformed data."""
        return self.df


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Kit")

    parser.add_argument("--data", required=True, help="Input CSV")
    parser.add_argument("--output", "-o", required=True, help="Output CSV")

    parser.add_argument("--encode", action="store_true", help="Encode categoricals")
    parser.add_argument("--scale", choices=['standard', 'minmax'], help="Scale features")
    parser.add_argument("--impute", choices=['mean', 'median'], help="Impute missing")

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded data: {df.shape}")

    kit = FeatureEngineeringKit()
    kit.load_data(df)

    if args.encode:
        print("Encoding categorical features...")
        kit.encode_categorical()

    if args.impute:
        print(f"Imputing missing values ({args.impute})...")
        kit.handle_missing(strategy=args.impute)

    if args.scale:
        print(f"Scaling features ({args.scale})...")
        kit.scale_features(method=args.scale)

    result = kit.get_data()
    print(f"Engineered data: {result.shape}")

    result.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
