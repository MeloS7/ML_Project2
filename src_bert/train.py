"""
Finetuning BERT model. 

Please check for:
    - `model_path` where to save trained model.
    - `train_path` and `test_path` to find data
"""


########################################################
# ---------------- parameters: PATHS ----------------- #
########################################################
train_path = "../data/split/partial/train.tsv"
test_path = "../data/split/partial/test.tsv"
model_path = "../models/bert_model.h5"



import numpy as np
import transformers as ppb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from build_model import DNN, BERTModel, build_model
from dataloader import DatasetBERT

TENSOR_DTYPE = torch.float32
torch.set_default_dtype(TENSOR_DTYPE)
print("Pytorch version:", torch.__version__, end=" | ")
print("gpu available:", torch.cuda.is_available(), ",", torch.cuda.get_device_name(0))
print(torch.cuda.device_count())


########################################################
# ------------------- build Model -------------------- #
########################################################
# Load pre-trained BERT base
bert_base = ppb.DistilBertModel.from_pretrained('distilbert-base-uncased').cuda()

top_clf = DNN(input_size=768)
bert_model = BERTModel(bert_base, top_clf)

########################################################
# -------------------- Load Data --------------------- #
########################################################
dataset_tr = DatasetBERT(train_path)
dataset_te = DatasetBERT(test_path)


########################################################
# -------------- train only top_clf ------------------ #
########################################################
np.random.seed(114514)

# Parameters
lr = 5e-5
batch_size = 64
maxiter = 2000
validation_per = 100
optimizer_train = optim.RMSprop(top_clf.parameters(), lr=lr)
scheduler = None #ExponentialLR(optimizer_train, gamma=0.999)

# Dataloader
train_loader = DataLoader(dataset_tr, batch_size=batch_size)#, num_workers=2)
valid_loader = DataLoader(dataset_te, batch_size=batch_size)#, num_workers=2)

# Start training
print("\n---------- Train top-classifier ----------\n")
results_train = build_model(bert_model, 
    lr = lr,
    train_loader = train_loader,
    test_loader = valid_loader,
    criterion = F.mse_loss,
    optimizer = optimizer_train,
    max_iter = maxiter,
    validate_per = validation_per,
    scheduler = scheduler
)


########################################################
# -------------------- finetuning -------------------- #
########################################################
np.random.seed(114514)

class LinearLR():
    """Customized linear scheduler"""
    def __init__(self, optimizer, init_lr, total_iters=5):
        self.optimizer = optimizer
        self.lr_decay = init_lr / total_iters
    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] -= self.lr_decay


# Parameters
lr = 6e-6
batch_size = 64
maxiter = 2000
validation_per = 100
optimizer_finetune = optim.AdamW(bert_model.parameters(), lr=lr, weight_decay=3e-3)
scheduler = LinearLR(optimizer_finetune, init_lr=lr, total_iters=maxiter)

# Dataloader
train_loader = DataLoader(dataset_tr, batch_size=batch_size)#, num_workers=2)
valid_loader = DataLoader(dataset_te, batch_size=batch_size)#, num_workers=2)

# Start training
print("\n---------- Fine-tuning ----------\n")
results_finetune = build_model(bert_model, 
    lr = lr,
    train_loader = train_loader,
    test_loader = valid_loader,
    criterion = F.mse_loss,
    optimizer = optimizer_finetune,
    max_iter = maxiter,
    validate_per = validation_per,
    scheduler = scheduler
)

torch.save(bert_model, model_path)
print("\n---------- finished ----------\n")