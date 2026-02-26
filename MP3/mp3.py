# Starter code for DSC 240 MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    X["AGE_YEARS"] = (-pd.to_numeric(X["DAYS_BIRTH"], errors="coerce")) / 365.25

    days_emp = pd.to_numeric(X["DAYS_EMPLOYED"], errors="coerce")
    X["UNEMPLOYED_FLAG"] = (days_emp > 0).astype(int)
    X["EMPLOYED_YEARS"] = np.where(days_emp < 0, (-days_emp) / 365.25, 0.0)

    inc = pd.to_numeric(X["AMT_INCOME_TOTAL"], errors="coerce")
    X["LOG_INCOME"] = np.log1p(np.maximum(0, inc))

    X["INCOME_PER_CHILD"] = inc / (1.0 + pd.to_numeric(X["CNT_CHILDREN"], errors="coerce").fillna(0))
    X["INCOME_PER_FAM"] = inc / (1.0 + pd.to_numeric(X["CNT_FAM_MEMBERS"], errors="coerce").fillna(0))
    X["EMPLOY_AGE_RATIO"] = X["EMPLOYED_YEARS"] / (X["AGE_YEARS"] + 1e-6)
    X["LOGINC_X_EMPLOY"] = X["LOG_INCOME"] * X["EMPLOYED_YEARS"]

    return X

def build_pipeline(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), selector(dtype_include=np.number)),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    scale_pos_weight = (neg / max(pos, 1))

    clf = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=1,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            reg_alpha=0.0,
            gamma=0.0,
            max_delta_step=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=0,
            n_jobs=-1
        )

    pipe = Pipeline(steps=[
        ("feat", FunctionTransformer(feature_engineer, validate=False)),
        ("preprocess", preprocess),
        ("clf", clf),
    ])
    return pipe

def f1_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    qs = np.linspace(0.0, 1.0, 2001)
    cand = np.unique(np.quantile(prob_pos, qs))

    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        pred = (prob_pos >= t).astype(int)

        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    print(f"Best F1: {best_f1:.4f} at threshold {best_t:.4f}")
    return best_t

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    train = training_data.copy()
    y = train["target"].astype(int).values
    X = train.drop(columns=["target", "QUANTIZED_INC", "QUANTIZED_AGE", "QUANTIZED_WORK_YEAR"])

    test = testing_data.copy()
    X_test = test.drop(columns=["QUANTIZED_INC", "QUANTIZED_AGE", "QUANTIZED_WORK_YEAR"])

    pipe = build_pipeline(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thresh = f1_threshold(y, oof_prob)

    pipe.fit(X, y)
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= thresh).astype(int)
    return test_pred


if __name__ == '__main__':

    training = pd.read_csv('./data/train.csv')
    development = pd.read_csv('./data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)

    


    


