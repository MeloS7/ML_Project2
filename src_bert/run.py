"""
Produce `sample_submission.csv` file.

pretrained BERT model is used. 
"""

model_path = "../models/bert_model.h5"          # path to BERT model
test_path = "../data/test_data.txt"             # path to test data
result_path = "../data/sample_submission.csv"   # to store results for submission


import pandas as pd
import torch
import transformers as ppb
import tqdm
from helpers import load_test_data, create_csv_submission, make_prediction
from preprocess import basic_clean_tweet
from dataloader import DatasetBERT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load model
print("Loading model ...")
bert_model = torch.load(model_path).to(device=device)

# Load data
print("Loading data ...")
df = load_test_data(test_path)

# preprocess data
print("Cleaning data ...")
df['tweet'] = df['tweet'].apply(basic_clean_tweet)

# make prediction
print("Forward pass ...")
for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    features = DatasetBERT.preprocess(row['tweet']).unsqueeze(0)
    features = features.to(device=device)
    pred = bert_model(features)
    pred = make_prediction(pred, zero_one=False).item()
    df.loc[idx, 'Prediction'] = pred

# save data
print("Saving results")
create_csv_submission(df["Id"], df["Prediction"], result_path)

print("Finished. ")