import numpy as np
from os import path
import sys
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from baseline_method.load_data import load_dataset
from method.lda_model import LdaTools
from method.train import train
from method.test import test
def run_model(train_perc=0.7, shuffle=True):
    # prepare data
    x, y = load_dataset()
    if shuffle:
        idx = np.random.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]
    train_size = int(x.shape[0] * train_perc)
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:]
    y_test = y[train_size:]

    # train lda
    lda_tool = LdaTools()
    lda_tool.train_lda()
    lda_tool.show_topics(10, 10)

    # train model
    net, sita = train(x_train, y_train, lda_tool)

    # test model
    test(x_test, y_test, net)



if __name__ == '__main__':
    run_model()