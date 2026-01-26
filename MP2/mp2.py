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
    return random.choices([0, 1], k=len(testing_data))

    #TODO implement your model and return the prediction

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


    


