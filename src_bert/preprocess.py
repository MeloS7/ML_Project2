"""
What is this file doing ?
1. perform cleaning on the tweets. 
    - `basic_clean_tweet` is used by default.
    - ambiguous tweets are removed (same tweets appear with both labels) 
2. Then merge positive and negative tweets (after assigning labels)
3. Store datasets into train / test tsv files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

def basic_clean_tweet(tweet:str):
    """
    Basic cleaning on tweet:
        - remove meaningless html tags;
        - replace consequent white spaces by single space ' '.
    """
    # clean html tags
    tweet = re.sub(r'</?\w+>', ' ', tweet)
    # reduce consequent white spaces to single space ' '
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    return tweet


def basic_clean_dataset(ser_pos, ser_neg, clean_method):
    """
    Basic cleaning on training data set of tweets: 
        - remove duplicates;
        - ambiguous tweets are removed (same tweets appear with both labels)
    Args:
        ser: pd.Series of tweets to clean
        clean_method: Callable[str, [str]] tweet cleaning function
    Returns:
        cleaned texts
    """

    # clean tweets, drop duplicates, empty tweet
    for ser in [ser_pos, ser_neg]:
        # clean tweets
        ser[:] = ser.map(clean_method)

        # drop duplicates
        ser.drop_duplicates(inplace=True)

        # drop rows with empty text
        ser.replace('', np.nan, inplace=True)
        ser.dropna(inplace=True)

    # ambiguous tweets:
    ambiguous = set(ser_pos) & set(ser_neg)

    for ser in [ser_pos, ser_neg]:
        # drop ambiguous tweets
        ser[:] = ser.map(lambda x: np.nan if x in ambiguous else x)
        ser.dropna(inplace=True)

    return ser_pos, ser_neg


##########################  MAIN  #######################
if __name__ == "__main__":

    # ------------ process parameters -------------- #
    test_ratio = 0.05
    path_data = '../data/'
    path_pos = path_data + "train_pos.txt"
    path_neg = path_data + "train_neg.txt"
    path_train = path_data + "split/partial/train.tsv"
    path_test = path_data + "split/partial/test.tsv"
    clean_method = basic_clean_tweet
    
    # ------------ process begins -------------- #
    # load file
    print("Loading files ...")
    with open(path_pos, 'r', encoding='utf-8') as f:
        ser_pos = pd.Series(f.read().splitlines())
    with open(path_neg, 'r', encoding='utf-8') as f:
        ser_neg = pd.Series(f.read().splitlines())

    # clean
    print("Cleaning ...")
    ser_pos, ser_neg = basic_clean_dataset(ser_pos, ser_neg, clean_method)

    # insert label
    print("Creating labels ...")
    df_pos = pd.DataFrame({
        'label': np.ones(ser_pos.shape),
        'tweet': ser_pos
    })
    df_neg = pd.DataFrame({
        'label': np.zeros(ser_neg.shape),
        'tweet': ser_neg
    })

    # split data
    print("Splitting ...")
    pos_tr, pos_te = train_test_split(df_pos, test_size=test_ratio)
    neg_tr, neg_te = train_test_split(df_neg, test_size=test_ratio)
    # create train / test df
    df_tr = pd.concat([pos_tr, neg_tr], axis=0)
    df_te = pd.concat([pos_te, neg_te], axis=0)

    # save to paths
    print("Saving to files ...")
    df_tr.to_csv(path_train, sep='\t', index=False)
    df_te.to_csv(path_test, sep='\t', index=False)

    print("Finished. ")
    