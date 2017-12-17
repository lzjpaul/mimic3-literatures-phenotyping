import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import sys
from os import path
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from method.define_mlp import Net
from baseline_method.compute_accurency import get_macro_micro_auc, get_auc_list

def test(x_test, y_test, net,
          batchsize=10,):
    test_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(x_test),
                                      target_tensor=torch.from_numpy(y_test))
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=2)
    res = []
    for data_iter in test_loader:
        tx, ty = data_iter
        outputs = net(Variable(tx).float())
        predicted = outputs.data
        res.extend(list(predicted.numpy()))
    auc_list, _ = get_auc_list(y_test, res)
    print'AUC List:'
    print auc_list