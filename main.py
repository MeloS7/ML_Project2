import sys
sys.path.append('./src')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loading import read_train_data
from pre_processing import Pre_processing
from vectorization import Vectorization
from sklearn.svm import SVC


# Data Loading
train_data, train_y = read_train_data(True, Batch_number=20000)

# Split training data
X_train, X_test, y_train, y_test = train_test_split(train_data, train_y, test_size=0.25, random_state=42)

# Data Pre-Processing
preproc_train = Pre_processing(X_train)
preproc_test = Pre_processing(X_test)
train_corpus, train_doc = preproc_train.streamline()
test_corpus, test_doc = preproc_test.streamline()

# Vectorization
vect_method = Vectorization(train_corpus, train_doc, test_doc)
train_X_vect, test_X_vect = vect_method.N_gram(n=2)

# Train Model
print("======= Start Training =======")
model = SVC(C=0.5, kernel='linear')
model.fit(train_X_vect, y_train)
print("======= END =========")

# Test
y_pred_train = model.predict(train_X_vect)
y_pred_test = model.predict(test_X_vect)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f'The accuracy of training data prediction is : {acc_train}')
print(f'The accuracy of test data prediction is : {acc_test}')
