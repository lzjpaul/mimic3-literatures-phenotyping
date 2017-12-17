from os import path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.sigmod = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmod(out)
        out = self.fc2(out)
        out = self.sigmod(out)
        return out


