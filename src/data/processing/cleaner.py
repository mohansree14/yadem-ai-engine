"""
YADEM Data Cleaner & Preprocessor
=============================================================================
Handles outlier detection, missing data imputation, format standardization,
and class balancing (SMOTE) before data enters the feature engineering stage.
Corresponds to Step 6 of the YADEM data acquisition pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Optional, Dict
from loguru import logger


class DataCleaner:
    """
    Cleans and preprocesses raw applicant data for the YADEM engine.
    Performs validation, cleaning, imputation, encoding, and scaling.
    """

    # Columns that should never be used as model features
    EXCLUDED_COLUMNS = [
        "applicant_id", "_prob_default", "default",
        "business_state", "business_zone",
    ]

    # Categorical columns requiring encoding
    CATEGORICAL_COLUMNS = [
        "business_sector", "owner_gender", "owner_education",
    ]

    # Numeric columns that should be log-transformed for normality
    LOG_TRANSFORM_COLUMNS = [
        "avg_monthly_revenue_6m", "avg_monthly_revenue_3m",
        "total_inflows_6m", "total_outflows_6m",
        "avg_monthly_balance", "requested_loan_amount_ngn",
        "mobile_money_monthly_volume", "pos_monthly_volume_ngn",
        "total_outstanding_debt_ngn", "avg_monthly_utility_spend_ngn",
        "avg_monthly_recharge_ngn",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.feature_columns: List[str] = []
        self._is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "default",
        apply_smote: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Fit preprocessors and transform the training data.
        
        Returns:
            (X_processed, y, feature_columns)
        """
        logger.info(f"Preprocessing data: {df.shape[0]} records, {df.shape[1]} columns")

        df_clean = df.copy()

        # Step 1: Handle missing values
        df_clean = self._handle_missing(df_clean)

        # Step 2: Remove outliers
        df_clean = self._remove_outliers(df_clean)

        # Step 3: Encode categorical variables
        df_clean = self._encode_categoricals(df_clean, fit=True)

        # Step 4: Log-transform skewed columns
        df_clean = self._log_transform(df_clean)

        # Step 5: Separate features and target
        y = df_clean[target_col].astype(int)
        X = df_clean.drop(
            columns=[c for c in self.EXCLUDED_COLUMNS if c in df_clean.columns],
            errors="ignore"
        )

        # Step 6: Identify numeric feature columns
        self.feature_columns = [
            c for c in X.columns
            if X[c].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]
        ]
        X = X[self.feature_columns]

        # Step 7: Impute remaining NaN
        X_values = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_values, columns=self.feature_columns, index=X.index)

        # Step 8: Scale features
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

        # Step 9: Apply SMOTE for class balance (if training)
        if apply_smote and y.mean() < 0.30:
            logger.info(
                f"Applying SMOTE: pre-balance default rate = {y.mean():.2%}"
            )
            smote = SMOTE(random_state=42, sampling_strategy=0.35)
            X_values, y_values = smote.fit_resample(X.values, y.values)
            X = pd.DataFrame(X_values, columns=self.feature_columns)
            y = pd.Series(y_values, name=target_col)
            logger.info(
                f"Post-SMOTE: {len(X)} records, default rate = {y.mean():.2%}"
            )

        self._is_fitted = True
        logger.info(
            f"Preprocessing complete: {X.shape[0]} records, "
            f"{X.shape[1]} features"
        )
        return X, y, self.feature_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors (inference time)."""
        if not self._is_fitted:
            raise RuntimeError("DataCleaner must be fitted first. Call fit_transform().")

        df_clean = df.copy()
        df_clean = self._handle_missing(df_clean)
        df_clean = self._encode_categoricals(df_clean, fit=False)
        df_clean = self._log_transform(df_clean)

        X = df_clean[[c for c in self.feature_columns if c in df_clean.columns]]

        # Add missing columns with zeros
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]
        X_values = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_values)

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with domain-appropriate strategies."""
        # Fill categorical NaN
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        # Fill numeric NaN with 0 for binary/count columns
        binary_cols = [c for c in df.columns if c.startswith(("has_", "is_", "bvn_", "nin_", "cac_"))]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _remove_outliers(
        self, df: pd.DataFrame, z_threshold: float = 4.0
    ) -> pd.DataFrame:
        """Remove extreme outliers using z-score method (conservative threshold)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = [c for c in self.EXCLUDED_COLUMNS if c in numeric_cols]
        check_cols = [c for c in numeric_cols if c not in exclude]

        initial_count = len(df)
        for col in check_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                z_scores = np.abs((df[col] - col_mean) / col_std)
                df = df[z_scores < z_threshold]

        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outlier records ({removed/initial_count:.1%})")

        return df.reset_index(drop=True)

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Label-encode categorical columns."""
        for col in self.CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen labels gracefully
                    df[col] = df[col].astype(str).map(
                        lambda x, _le=le: (
                            _le.transform([x])[0]
                            if x in _le.classes_
                            else -1
                        )
                    )
        return df

    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p transform to highly skewed numeric columns."""
        for col in self.LOG_TRANSFORM_COLUMNS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df
