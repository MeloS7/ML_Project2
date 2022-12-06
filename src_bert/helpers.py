import csv
import numpy as np
import pandas as pd
import torch
from preprocess import basic_clean_tweet


"""
    File manipulation
"""
def load_test_data(test_path):
    """ Load id,tweet to pandas.DataFrame
    """
    df = pd.DataFrame(columns=['Id','tweet'])
    with open(test_path, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            df.loc[idx] = line.split(',', 1)
    return df


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

"""
    Model Evaluation
"""
@torch.no_grad()
def eval_accuracy(pred, y_true):
    """
    Args:
        pred array-like, containing floating numbers in (0,1)
    """
    pred_labels = (pred > 0.5).to(y_true.dtype)
    return (torch.sum(pred_labels == y_true) / pred.size(dim=0)).item()

@torch.no_grad()
def make_prediction(pred, zero_one=True):
    """ map floating value between 0 and 1 to labels in {0,1} or {-1,1}
    Args:
        pred: torch.tensor
        zero_one: map into {0,1} if set to True else into {-1,1}
    """
    pred = (pred > 0.5).to(torch.int32)
    if not zero_one:
        pred = 2 * pred - 1
    return pred.to(torch.int32)