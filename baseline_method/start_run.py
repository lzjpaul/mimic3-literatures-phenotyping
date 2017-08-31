import pandas as pd
import numpy as np
from load_data import load_corpus
from baseline_method.multi_logistic_model import MultiLogistic
from utility.csv_utility import CsvUtility

if __name__ == '__main__':
    print 'loading data...'
    train_x, train_y, test_x, test_y, idx = load_corpus()
    print 'loading ready...'
    multi_logs = MultiLogistic(len(train_y[0]))
    print 'training...'
    multi_logs.training(training_x=train_x, training_y=train_y)
    print 'testing...'
    re_auc, re_list = multi_logs.testing(testing_x=test_x, testing_y=test_y)
    print re_auc[:-1]
    print re_auc[-1]
    CsvUtility.write2pickle('../data-repository/model_multilogisticCV.pickle', [idx, re_auc, re_list], 'w')