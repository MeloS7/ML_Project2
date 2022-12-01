import os
import numpy as np

train_neg_path   = r'./data/twitter-datasets/train_neg.txt'
train_neg_full_path = r'./data/twitter-datasets/train_neg_full.txt'
train_pos_path   = r'./data/twitter-datasets/train_pos.txt'
train_pos_full_path = r'./data/twitter-datasets/train_pos_full.txt'
test_path = r'./data/twitter-datasets/test_data.txt'

def read_text(filepath):
    contents = []
    try:
        with open(filepath, "r") as f:
            for line in f.readlines():
                contents.append(line) 
    except Exception as erro:
        print("Error with info :" + erro)
    return contents


def read_train_data(IF_BATCH=True, Batch_number=20000):
    train_neg_data = read_text(train_neg_path)
    train_pos_data = read_text(train_pos_path)
    train_y_neg = np.zeros(len(train_neg_data)) - 1
    train_y_pos = np.ones(len(train_pos_data))

    if not IF_BATCH:
        Batch_number = len(train_neg_data)

    train_x = train_neg_data[:Batch_number] + train_pos_data[:Batch_number]
    train_y = np.concatenate((train_y_neg[:Batch_number], train_y_pos[:Batch_number]), axis=0)

    return train_x, train_y
        
        