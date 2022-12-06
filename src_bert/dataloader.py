import torch
import numpy as np
import pandas as pd
import transformers as ppb
from torch.utils.data import Dataset


class DatasetBERT(Dataset):
    """
    Dataset for BERT model. 
    """
    # Load BERT tokenizer for preprocessing
    tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # length of result tweets when padding / truncating
    max_len = 60
    
    def __init__(self, data_path):
        """
        Args:
            data_path:  path to tsv file containing (label, text) in each row. 
        """
        # create datasets and shuffle index
        self.df = pd.read_csv(data_path, sep='\t')
        # shuffle index for retrieving
        self.indexmap = np.random.permutation(np.arange(len(self.df)))

    @staticmethod
    def preprocess(text):
        """ convert sentance to faeture tensor """
        features = DatasetBERT.tokenizer.encode(text, add_special_tokens=True, 
            padding="max_length", truncation=True, max_length=DatasetBERT.max_len)
        return torch.tensor(features, dtype=torch.int32)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Args:
            index: index of data to retrieve
        Returns:
            (tweet, label)
        """
        index = self.indexmap[index]
        return DatasetBERT.preprocess(self.df.loc[index, "tweet"]), \
               torch.tensor(self.df.loc[index, 'label'], dtype=torch.float32)
    