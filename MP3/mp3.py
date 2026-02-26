# Starter code for DSC 240 MP3 - upgraded solution (sklearn only)
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

np.random.seed(0)

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # quantized columns listed in data.md :contentReference[oaicite:2]{index=2}
        self.drop_cols = ["QUANTIZED_INC", "QUANTIZED_AGE", "QUANTIZED_WORK_YEAR"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # ---- drop all quantized columns (if present) ----
        for c in self.drop_cols:
            if c in X.columns:
                X = X.drop(columns=[c])

        # ---- base numeric columns ----
        income = X["AMT_INCOME_TOTAL"].astype(float) if "AMT_INCOME_TOTAL" in X.columns else pd.Series(np.nan, index=X.index)
        days_birth = X["DAYS_BIRTH"].astype(float) if "DAYS_BIRTH" in X.columns else pd.Series(np.nan, index=X.index)
        days_employed = X["DAYS_EMPLOYED"].astype(float) if "DAYS_EMPLOYED" in X.columns else pd.Series(np.nan, index=X.index)

        cnt_children = X["CNT_CHILDREN"].astype(float) if "CNT_CHILDREN" in X.columns else pd.Series(np.nan, index=X.index)
        fam_members = X["CNT_FAM_MEMBERS"].astype(float) if "CNT_FAM_MEMBERS" in X.columns else pd.Series(np.nan, index=X.index)

        # ---- add: age_years ----
        # DAYS_BIRTH counts backwards from 0, so typically negative; convert to positive years :contentReference[oaicite:3]{index=3}
        X["age_years"] = (-days_birth / 365.25)

        # ---- add: unemployed_flag (categorical) ----
        # DAYS_EMPLOYED positive => currently unemployed :contentReference[oaicite:4]{index=4}
        unemployed = (days_employed > 0)
        X["unemployed_flag"] = np.where(unemployed, "Y", "N").astype(object)

        # ---- add: employed_years ----
        # if unemployed, set 0; else convert negative DAYS_EMPLOYED to positive years
        employed_years = np.where(days_employed < 0, (-days_employed / 365.25), 0.0)
        X["employed_years"] = employed_years

        # ---- add: log_income ----
        X["log_income"] = np.log1p(np.maximum(income, 0.0))

        # ---- interactions ----
        # income per child (avoid divide-by-zero): income / (children + 1)
        X["income_per_child"] = income / (np.maximum(cnt_children, 0.0) + 1.0)

        # income per family member (avoid divide-by-zero): income / max(fam_members, 1)
        X["income_per_family_member"] = income / np.maximum(fam_members, 1.0)

        # employed_years to age ratio (avoid divide-by-zero)
        X["employed_to_age_ratio"] = X["employed_years"] / np.maximum(X["age_years"], 1e-6)

        # log_income to employed_years interaction
        X["log_income_x_employed_years"] = X["log_income"] * X["employed_years"]

        return X

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1 - labels[expected == 0])
    fn = np.sum(1 - labels[expected == 1])
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    error_rate = (fp + fn) / (tp + fp + tn + fn + 1e-12)
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


class CreditRiskF1Classifier(BaseEstimator, ClassifierMixin):
    """
    A custom sklearn-style classifier:
      - robust preprocessing (impute + onehot + scale)
      - HistGradientBoosting for strong tabular performance
      - threshold tuning to maximize F1 on out-of-fold predictions
    """
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 0,
        threshold_grid_size: int = 401,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.threshold_grid_size = threshold_grid_size
        self.model_params = model_params

        self.pipeline_: Optional[Pipeline] = None
        self.threshold_: float = 0.5
        self.feature_columns_: Optional[List[str]] = None

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        fe = FeatureEngineer()
        X_fe = fe.transform(X)

        # Infer column types
        cols = list(X_fe.columns)

        # Treat "object" or "category" as categorical.
        # Everything numeric is treated as numeric (includes 0/1 flags).
        cat_cols = [c for c in cols if str(X_fe[c].dtype) in ("object", "category")]
        num_cols = [c for c in cols if c not in cat_cols]

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # safe w/ sparse combos
        ])

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        # Default model params tuned for tabular + imbalance
        params = dict(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=500,
            max_leaf_nodes=31,
            max_depth=None,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
        )
        if self.model_params:
            params.update(self.model_params)

        clf = HistGradientBoostingClassifier(**params)

        return Pipeline(steps=[
            ("fe", fe),
            ("pre", pre),
            ("clf", clf),
        ])

    @staticmethod
    def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
        # simple inverse-frequency weighting
        y = y.astype(int)
        n = len(y)
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        return np.where(y == 1, w_pos, w_neg).astype(float)

    def _tune_threshold(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        # Search threshold that maximizes F1 on positive class (=1)
        # Use a dense grid; includes typical imbalanced sweet spots.
        thresholds = np.linspace(0.01, 0.99, self.threshold_grid_size)
        best_t = 0.5
        best_f1 = -1.0
        for t in thresholds:
            preds = (proba >= t).astype(int)
            f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X = X.copy()
        self.feature_columns_ = list(X.columns)

        # Build pipeline
        self.pipeline_ = self._build_pipeline(X)

        # Out-of-fold probabilities for threshold tuning
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(X), dtype=float)

        y = np.asarray(y).astype(int)
        sample_weight = self._balanced_sample_weight(y)

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            w_tr = sample_weight[tr_idx]

            pipe = self._build_pipeline(X_tr)
            pipe.fit(X_tr, y_tr, clf__sample_weight=w_tr)

            # HistGB supports predict_proba
            p = pipe.predict_proba(X_va)[:, 1]
            oof_proba[va_idx] = p

        # Tune threshold using OOF preds (less overfit than a single split)
        self.threshold_ = self._tune_threshold(y, oof_proba)

        # Fit final model on all data
        self.pipeline_.fit(X, y, clf__sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Model not fit yet.")
        # Align columns defensively
        X = X.copy()
        if self.feature_columns_ is not None:
            missing = [c for c in self.feature_columns_ if c not in X.columns]
            for c in missing:
                X[c] = np.nan
            X = X[self.feature_columns_]

        proba = self.pipeline_.predict_proba(X)[:, 1]
        return (proba >= self.threshold_).astype(int)


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Train on training_data (includes target) and return predictions on testing_data.
    Keep this signature unchanged (Gradescope requirement).
    """
    train = training_data.copy()
    y = train["target"].values.astype(int)
    X = train.drop(columns=["target"])

    model = CreditRiskF1Classifier(
        n_splits=5,
        random_state=0,
        threshold_grid_size=401,
        model_params=None,  # you can tweak if you want
    )
    model.fit(X, y)

    preds = model.predict(testing_data).astype(int)
    return preds.tolist()


if __name__ == "__main__":
    training = pd.read_csv("./data/train.csv")
    development = pd.read_csv("./data/dev.csv")

    target_label = development["target"].values.astype(int)
    development = development.drop("target", axis=1)

    prediction = np.array(run_train_test(training, development), dtype=int)
    status = compute_metric(prediction, target_label)
    print(status)