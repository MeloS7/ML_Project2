import os
import pandas as pd

def save_data_fasttext(data, if_train=True):
    tofiledir='./data/twitter-datasets'
    if if_train:
        filename = '/train_fast.txt'
    else:
        filename = '/test_fast.txt'

    with open(tofiledir + filename,'w') as f:
        for s in data:
            f.writelines(s + '\n')

def deal_pre(data, method_t):
	# Convert list to dataframe
    df = pd.DataFrame(data, columns=['Prediction'])
    df.index = df.index + 1
    df.insert(0, 'Id', range(1, len(df)+1), allow_duplicates=False)
    
	# Save scv file to local
    filename = method_t+"_sub.csv"
    df.to_csv("./data/twitter-datasets/submission/" + filename, index=False)
    return df

def deal_fasttext(test_doc):
    tofiledir='./data/twitter-datasets/trained_data'
    with open(tofiledir+'/test_fast.txt','w') as f:
        for s in test_doc:
            f.writelines(s + '\n')
    
    df = pd.read_fwf(tofiledir+'/test_fast.txt', header=None)
    df.columns=["raw_text", "c2", "c3"]
    df = df.drop(columns=['c2', 'c3'])
    
    return df