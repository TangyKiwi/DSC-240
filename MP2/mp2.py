# Starter code for DSC 240 MP2
import math
import random
import numpy as np
import pandas as pd

from typing import List

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: pd.DataFrame
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1], k=len(testing_data))
    """

    #TODO implement your model and return the prediction

    X_train = training_data.drop('target', axis=1).values
    y_train = training_data['target'].values # either 0 or 1

    # use perceptron from HW1
    w = np.zeros(training_data.shape[1]) # keep extra dim for bias
    max_i = 1000
    for i in range(max_i):
        converged = True
        for point, label in zip(X_train, y_train):
            x = np.hstack(([1], point))
            y = 1 if label == 1 else -1
            if y * np.dot(w, x) <= 0:
                w += y * x
                converged = False
        if converged:
            print(f"Converged after {i} iterations")
            break

    if i == max_i - 1:
        print("Did not converge")

    X_test = testing_data.values
    y_pred = []
    for point in X_test:
        x = np.hstack(([1], point))
        pred = 1 if np.dot(w, x) > 0 else 0
        y_pred.append(pred)

    return y_pred

if __name__ == '__main__':
    # load data
    training = pd.read_csv('data/train.csv')
    testing = pd.read_csv('data/dev.csv')
    target_label = testing['target']
    testing.drop('target', axis=1, inplace=True)

    # run training and testing
    prediction = run_train_test(training, testing)

    # Example metric 1: check accuracy 
    target_label = target_label.values
    print("Dev Accuracy: ", np.sum(prediction == target_label) / len(target_label))
    
    # Metric 2: F1 score
    # Please implement F1 score metric to test your predictions. We do not evlaute your F1 score function
    # nor do you need to provide it, but you should implement it for your understading. 
    # Please note: Autograder will evaluate your predictions on hidden test data using F1 scoring.

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for y_true, y_pred in zip(target_label, prediction):
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Dev F1 Score: ", f1_score)


    


