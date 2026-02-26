# Starter code for DSC 240 MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

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

    return X

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    binary_cols = [
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL",
        "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "UNEMLOYED_FLAG"
    ]

    numeric_cols = [
        "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED", "AGE_YEARS", 
        "EMPLOYED_YEARS", "LOG_INCOME"
    ]

    known = set(binary_cols + numeric_cols)
    categorical_cols = [col for col in X.columns if col not in known]

    binary_cols = [col for col in binary_cols if col in X.columns]
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]

    cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    preprocess = ColumnTransformer(
        transformers=[
            ("num",
             Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
             ]), numeric_cols),
            ("bin",
             Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=True)),
              ]), binary_cols),
            ("cat",
             Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", cat_ohe),
             ]), categorical_cols),
        ], 
        remainder="drop", 
        sparse_threshold=0.3
    )

    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=2.0,
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        random_state=0
    )

    pipe = Pipeline(steps=[
        ("feat", FunctionTransformer(feature_engineer, validate=False)),
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    return pipe

def f1_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    unique = np.unique(prob_pos)
    if len(unique) > 2000:
        qs = np.linspace(0.0, 1.0, 2001)
        cand = np.quantile(prob_pos, qs)
        cand = np.unique(cand)
    else:
        cand = unique

    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        pred = (prob_pos >= t).astype(int)
        metric = compute_metric(pred, y_true)
        if metric["f1"] > best_f1:
            best_f1 = metric["f1"]
            best_t = t

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
    X = train.drop(columns=["target"])

    pipe = build_pipeline(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold = f1_threshold(y, oof_prob)

    pipe.fit(X, y)
    test_prob = pipe.predict_proba(testing_data)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)    

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

    


    


