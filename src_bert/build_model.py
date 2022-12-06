import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helpers import eval_accuracy


class DNN(nn.Module):
    def __init__(self, input_size, output_size=1, seed=42):
        super(DNN, self).__init__()
        
        # params
        fc1_units = 64

        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, output_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class BERTModel(nn.Module):
    """
    pretrained BERT base + top-classifier
    """
    def __init__(self, bert_base, top_clf, dtype=torch.float32):
        super(BERTModel, self).__init__()
        self.dtype = dtype
        
        self.bert_base = bert_base
        self.top_clf = top_clf

    def forward(self, x):
        mask = torch.where(x != 0, 1, 0).to(dtype=torch.int32)
        x = x.to(dtype=torch.int32)
        x = self.bert_base(x, mask)[0][:,0,:]
        x = x.to(dtype=self.dtype)
        x = self.top_clf(x)
        return x.to(dtype=self.dtype)



def build_model(
    model,
    train_loader, 
    test_loader,
    lr = 1e-4,
    criterion = F.mse_loss,
    optimizer = None,
    max_iter = 1000,
    validate_per = 10,
    scheduler = None
):
    """
        Training process of the model. 
        gpu is assumed to be available. 
    """
    device = torch.device("cuda:0")

    model.to(device)
    if optimizer is None:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optimizer_to(optimizer, "cuda")

    # supervision results for every checkpoint i.e. every `validation_per` iterations
    iters = []
    loss_tr = []
    loss_te = []
    acc_tr = []
    acc_te = []

    # keep trace of training loss for every iteration (reinitialized every `validation_per` iter)
    iter_loss_tr = []
    iter_acc_tr = []

    # training data generator
    generator = iter(train_loader)
    
    for n_iter in range(1, max_iter+1):
        # ---------- Load next batch ---------- #
        try:
            x, y = next(generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator = iter(train_loader)
            x, y = next(generator)
            
        # ---------- Training ---------- #
        model.train()
        
        # use gpu if available
        x = x.to(device)
        y = y.to(device)
        model.to(device)

        # compute loss
        pred = model(x).reshape(y.shape)
        loss = criterion(pred, y)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # scheduler
        if scheduler is not None:
            scheduler.step()

        # keep metrics for supervision
        iter_loss_tr.append(loss.item())
        iter_acc_tr.append(eval_accuracy(pred.detach(), y))

            
        # ----------Evaluation / validation ---------- #
        if n_iter % validate_per == 0:
            model.eval()
            # evaluate previous training loss/acc
            step_loss_tr = sum(iter_loss_tr) / len(iter_loss_tr)
            step_acc_tr = sum(iter_acc_tr) / len(iter_acc_tr)
            
            # evaluate validation loss/acc
            step_loss_te, step_acc_te = evaluate(model, test_loader, criterion, maxiter=validate_per)

            iters.append(n_iter)
            loss_tr.append(step_loss_tr)
            loss_te.append(step_loss_te)
            acc_tr.append(step_acc_tr)
            acc_te.append(step_acc_te)

            print(f"iter:{n_iter:5d} \
                loss_tr:{step_loss_tr} acc_tr:{step_acc_tr} \
                loss_te:{step_loss_te} acc_te:{step_acc_te}")

            # reinitialize for next validation
            iter_loss_tr = []
            iter_acc_tr = []
    
    results = {
        "iter": iters,
        "l_tr": loss_tr,
        "l_te": loss_te,
        "acc_tr": acc_tr,
        "acc_te": acc_te
    }

    return results



def optimizer_to(optim, device:str):
    """ change device of optimizer 
    """
    if device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)



@torch.no_grad()
def evaluate(model, dataloader, criterion, maxiter=20, device=None):
    """ Compute loss and accuracy on data loaded from `dataloader`.
    """
    model.eval()
    if device is not None:
        model.to(device)
    else:
        device = next(model.parameters()).device
    loss,acc = 0,0
    for cnt, (x,y) in enumerate(dataloader):
        x,y = x.to(device),y.to(device)
        pred = model(x).reshape(y.shape)
        loss += criterion(pred, y).item()
        acc += eval_accuracy(pred, y)
        if maxiter is not None and cnt + 1 >= maxiter:
            break
    loss /= (cnt + 1)
    acc /= (cnt + 1)

    return loss, acc