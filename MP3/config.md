```python
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
```
Best F1: 0.4808 at threshold 0.7560
{'f1': np.float64(0.5405405405405406), 'accuracy': np.float64(0.9229607250755287), 'precision': np.float64(0.6122448979591837), 'tpr': np.float64(0.4838709677419355), 'fpr': np.float64(0.03166666666666667), 'error_rate': np.float64(0.0770392749244713)}
```python
    clf = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.0,
            gamma=0.0,
            max_delta_step=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=0,
            n_jobs=-1
        )
```
Best F1: 0.4776 at threshold 0.7052  
{'f1': np.float64(0.5128205128205128), 'accuracy': np.float64(0.9138972809667674), 'precision': np.float64(0.5454545454545454), 'tpr': np.float64(0.4838709677419355), 'fpr': np.float64(0.041666666666666664), 'error_rate': np.float64(0.08610271903323263)}
```python
    clf = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=0,
            n_jobs=-1
        )
```
Best F1: 0.4783 at threshold 0.7413  
{'f1': np.float64(0.5087719298245614), 'accuracy': np.float64(0.9154078549848943), 'precision': np.float64(0.5576923076923077), 'tpr': np.float64(0.46774193548387094), 'fpr': np.float64(0.03833333333333333), 'error_rate': np.float64(0.08459214501510574)}

```python
    clf = XGBClassifier(
            n_estimators=3500,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=3.0,
            gamma=0.5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=0.75*scale_pos_weight,
            random_state=0,
            n_jobs=-1
        )
```
Best F1: 0.4263 at threshold 0.6120  
{'f1': np.float64(0.4666666666666667), 'accuracy': np.float64(0.9033232628398792), 'precision': np.float64(0.4827586206896552), 'tpr': np.float64(0.45161290322580644), 'fpr': np.float64(0.05), 'error_rate': np.float64(0.09667673716012085)}

```python
    clf = XGBClassifier(
            n_estimators=3500,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.0,
            gamma=0.2,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=1.3*scale_pos_weight,
            random_state=0,
            n_jobs=-1
        )
```
Best F1: 0.4787 at threshold 0.7657  
{'f1': np.float64(0.5), 'accuracy': np.float64(0.9123867069486404), 'precision': np.float64(0.5370370370370371), 'tpr': np.float64(0.46774193548387094), 'fpr': np.float64(0.041666666666666664), 'error_rate': np.float64(0.08761329305135952)}