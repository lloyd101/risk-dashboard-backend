"""
Model training and risk computation utilities for the risk scoring backend.

This module encapsulates the logic for reading in applicant data,
training a logistic regression model, and computing new risk scores
with adjustable feature weights.  It is based on the synthetic data
generator included in the notebook prototype but can also read data
from a database in the future.

Note: scikit‑learn transforms are used for demonstration purposes.
When deploying in a production setting you may prefer to persist
pretrained model artefacts and feature pipelines, or use a more
sophisticated model (e.g., XGBoost).  The functions here run on
startup and store their state in module‑level variables for reuse.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


class RiskModel:
    """Encapsulates a logistic regression model for risk scoring."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data.copy()
        # If the data already contains a 'baseline_risk' column, drop it
        if 'baseline_risk' in self.df.columns:
            self.df = self.df.drop(columns=['baseline_risk'])
        # For reproducibility, set a fixed random state
        self.random_state = 42
        # Prepare target if present
        if 'target' in self.df.columns:
            self.target = self.df['target'].values
            self.df = self.df.drop(columns=['target'])
        else:
            self.target = None
        # Train model if target available
        self.model: Optional[Pipeline] = None
        if self.target is not None:
            self.train_model()
        else:
            # In case no target is provided we still build a pipeline for preprocessing
            self.model = self._build_untrained_model()
        # Cache processed data for inference
        self._prepare_processed_matrix()

    def _build_untrained_model(self) -> Pipeline:
        numeric_features = ['age', 'tenure', 'prior_claims', 'asset_value', 'geo_risk']
        categorical_features = ['credit_band', 'asset_type']
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )
        # Dummy logistic regression used for feature dimensionality, will not be trained
        clf = LogisticRegression(max_iter=1)
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    def train_model(self) -> None:
        """Train the logistic regression model on the dataset."""
        # Ensure target exists
        if self.target is None:
            raise ValueError("Cannot train model without target labels.")
        numeric_features = ['age', 'tenure', 'prior_claims', 'asset_value', 'geo_risk']
        categorical_features = ['credit_band', 'asset_type']
        X = self.df[numeric_features + categorical_features]
        y = self.target
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )
        clf = LogisticRegression(max_iter=1000)
        self.model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        self.model.fit(X_train, y_train)
        # Optionally compute baseline risk on entire dataset
        probs = self.model.predict_proba(X)[:, 1]
        self.df['baseline_risk'] = probs
        # Prepare processed matrix for later weight adjustments
        self._prepare_processed_matrix()

    def _prepare_processed_matrix(self) -> None:
        """Transform raw features to model input space and cache the results."""
        if self.model is None:
            # Build untrained pipeline for preprocessing
            self.model = self._build_untrained_model()
        # Compute processed feature matrix
        self.X_proc = self.model.named_steps['preprocessor'].fit_transform(
            self.df[
                ['age', 'tenure', 'prior_claims', 'asset_value', 'geo_risk', 'credit_band', 'asset_type']
            ]
        )
        # Cache classifier weights if trained
        if hasattr(self.model.named_steps['classifier'], 'coef_'):
            self.coefficients = self.model.named_steps['classifier'].coef_.ravel()
            self.intercept = self.model.named_steps['classifier'].intercept_[0]
        else:
            # If classifier not trained, set coefficients to zeros with appropriate length
            self.coefficients = np.zeros(self.X_proc.shape[1])
            self.intercept = 0.0
        # Index of the standardized age column in the processed matrix (first numeric feature)
        self.age_idx = 0

    def compute_risk(self, age_weight: float = 1.0) -> np.ndarray:
        """Compute risk scores adjusting the age weight factor.

        The weight scales the contribution of the age coefficient in the
        logistic regression model.  A weight of 1.0 leaves the model
        unchanged; values below 1.0 diminish the effect of age, while
        values above 1.0 amplify it.
        """
        # Compute baseline linear predictor
        linear = np.dot(self.X_proc, self.coefficients) + self.intercept
        age_contribution = self.coefficients[self.age_idx] * self.X_proc[:, self.age_idx]
        adjusted_linear = linear - age_contribution + age_contribution * age_weight
        return 1 / (1 + np.exp(-adjusted_linear))
