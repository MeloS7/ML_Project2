import sys
sys.path.append('./src_SVM_RF_Fasttext')
import warnings
warnings.filterwarnings('ignore')

import argparse

import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loading import read_train_data, read_test_data
from pre_processing import Pre_processing
from vectorization import Vectorization
from data_save import save_data_fasttext, deal_pre, deal_fasttext
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from decimal import Decimal

METHOD_TRAINING = ["SVM", "RF", "Fasttext"]
METHOD_WE = ["CV", "TF", "NGRAM", "PRE"]
BINARY = [True, False]

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-m","--method", type=str,
                help="Choose a training model",
                required=True,
                default= "SVM")

parser.add_argument("-bn","--batch_number", type=int,
                help="Input a batch number for training",
                default= 10000)
                
parser.add_argument("-full","--full", type=bool,
                help="Choose if use the full dataset",
                default= False)

parser.add_argument("-mwe","--method_word_embedding", type=str,
                help="Choose a word embedding method",
                default= "PRE")

# ex. {'method': arg1, 'batch_number': arg2...}
args = vars(parser.parse_args())

method_training = args["method"]
method_we = args["method_word_embedding"]
BATCH_NUMBER = args["batch_number"]
IF_FULL = args["full"]
assert method_training in METHOD_TRAINING
assert method_we in METHOD_WE
assert IF_FULL in BINARY


# Data Loading
train_data, train_label = read_train_data(True, Batch_number = BATCH_NUMBER, full = IF_FULL)

if method_training in ['SVM', 'RF']:
    # Data Pre-Processing
    preproc_train = Pre_processing(train_data)
    train_corpus, train_doc, train_sents = preproc_train.streamline()

    # Vectorization
    vect_method = Vectorization(train_corpus, train_doc, method_we)
    train_data_vect, model_vect = vect_method.select_by_name()
else: # In case method of Fasttext
    # Data Pre-Processing
    preproc_train = Pre_processing(train_data)
    train_data_vect = preproc_train.add_label()

# Split training data
X_train, X_test, y_train, y_test = train_test_split(train_data_vect, train_label, test_size=0.25, random_state=42)

# Train Model
print("======= Start Training =======")
if method_training == "SVM":
    model = LinearSVC(random_state=42, tol=1e-05)
    model.fit(X_train, y_train)
elif method_training == "RF":
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
else:
    save_data_fasttext(X_train, if_train=True)
    save_data_fasttext(X_test, if_train=False)
    model=fasttext.train_supervised('./data/twitter-datasets/train_fast.txt', wordNgrams=5, epoch=5)
print("======= END =========")

# Test
if method_training in ['SVM', 'RF']:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f'The accuracy of training data prediction is : {acc_train}')
    print(f'The accuracy of test data prediction is : {acc_test}')
else:
    res_train = model.test('./data/twitter-datasets/train_fast.txt')
    res_test = model.test('./data/twitter-datasets/test_fast.txt')
    print(f'The accuracy of training data prediction is : {res_train[1]}')
    print(f'The accuracy of test data prediction is : {res_test[1]}')

# Output for submission
test_data = read_test_data()

if method_training in ["SVM", "RF"]:
    # Pre-Processing for test data
    preproc_test = Pre_processing(test_data)
    test_corpus, test_doc, test_sents = preproc_test.streamline(for_test=True)

    # Vectorization and the rest operations for test data
    if method_we == "PRE":
        test_embeddings = model_vect.encode(test_doc)
    else:
        test_embeddings = model_vect.transform(test_doc)

    y_output = model.predict(test_embeddings)
    df_output = deal_pre(y_output, method_training)
else:
    preproc_test = Pre_processing(test_data)
    preproc_test.add_label(for_test=True)
    test_doc = preproc_test.create_document()
    df = deal_fasttext(test_doc)

    def predict_f(row):
        return model.predict(row['raw_text'])
    df['predictions'] = df.apply(predict_f,axis=1)

    for index, row in df.iterrows():
        if df['predictions'][index][0] == ('__label__neg',):
            df.loc[index, "Prediction"] = Decimal(-1).to_integral()
        else:
            df.loc[index, "Prediction"] = Decimal(1).to_integral()
    
    df_output = df.drop(columns=['raw_text', 'predictions'])
    df_output.index += 1
    df_output.insert(0, 'Id', range(1, len(df_output)+1), allow_duplicates=False)
    df_output.to_csv("./data/twitter-datasets/submission/fasttext_sub.csv", index=False)
    



    

