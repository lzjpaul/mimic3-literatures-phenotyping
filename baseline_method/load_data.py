#!/usr/bin/env Python
# coding=utf-8

import cPickle as pickle
import pandas as pd
import numpy as np
import os
import math
from random import randrange, shuffle
import json
import datetime
from collections import defaultdict
from utility.csv_utility import CsvUtility
from utility.nlp_utility import NLP_Utility
from utility.double_dict_utility import DictDoubleMap


def get_dataset(data_pickle_path, word_dict_path, predict_dict_path):

    all_events = CsvUtility.read_pickle(data_pickle_path, 'r')
    word_dict = CsvUtility.read_pickle(word_dict_path, 'r')
    predict_dict = CsvUtility.read_pickle(predict_dict_path, 'r')
    print all_events[0]
    print len(word_dict), len(predict_dict)

    feature_dict = DictDoubleMap(list(word_dict))
    pred_dict = DictDoubleMap(list(predict_dict))

    feature_matrix = np.zeros((len(all_events), len(word_dict)))
    result_matrix = np.zeros((len(all_events), len(predict_dict)))

    for i_iter, event_line in enumerate(all_events):
        for event_item in event_line[0]:
            feature_matrix[i_iter][feature_dict.get_index_by_word(event_item)] += 1
        for pred_item in event_line[1]:
            result_matrix[i_iter][pred_dict.get_index_by_word(pred_item)] = 1

    return feature_matrix, result_matrix



def load_corpus(all_path='../data-repository/', train_perc=0.7):

    x, y = get_dataset(all_path+'after_instance.pickle',all_path+'event_instance_dict.pickle',
                all_path+'predict_diags_dict.pickle')
    train_size = int(x.shape[0] * train_perc)

    # shuffle the train set
    idx = np.random.permutation(x.shape[0])
    x_train = x[idx]
    y_train = y[idx]
    return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:], idx

if __name__ == '__main__':
    training_x, training_y, testing_x, testing_y = load_corpus()
