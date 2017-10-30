#!/usr/bin/env Python
# coding=utf-8

import numpy as np

from utility.csv_utility import CsvUtility
from utility.double_dict_utility import DictDoubleMap


def get_dataset(data_pickle_path, word_dict_path, predict_dict_path):
    all_events = CsvUtility.read_pickle(data_pickle_path, 'r')
    word_dict = CsvUtility.read_pickle(word_dict_path, 'r')
    predict_dict = CsvUtility.read_pickle(predict_dict_path, 'r')
    print all_events[0]
    print len(word_dict), len(predict_dict), len(all_events)

    feature_dict = DictDoubleMap(list(word_dict))
    pred_dict = DictDoubleMap(list(predict_dict))

    feature_matrix = np.zeros((len(all_events), len(word_dict)))
    result_matrix = np.zeros((len(all_events), len(predict_dict)))

    for i_iter, event_line in enumerate(all_events):
        for event_item in event_line[0]:
            feature_matrix[i_iter][feature_dict.get_index_by_word(event_item)] += 1
        for pred_item in event_line[1]:
            result_matrix[i_iter][pred_dict.get_index_by_word(pred_item)] = 1

        if i_iter % 1000 == 0:
            print 'complete {0} of {1}'.format(i_iter, len(all_events))

    CsvUtility.write_dict2csv(feature_dict.get_word2index(), '../data-repository/', 'feature2index.csv')
    CsvUtility.write_dict2csv(pred_dict.get_word2index(), '../data-repository/', 'predict2index.csv')
    CsvUtility.write_array2csv(feature_matrix, '../data-repository/', 'feature_matrix.csv')
    CsvUtility.write_array2csv(result_matrix, '../data-repository/', 'result_matrix.csv')

    return feature_matrix, result_matrix


def load_corpus(all_path='../data-repository/', train_perc=0.7):
    x, y = get_dataset(all_path + 'after_instance.pkl', all_path + 'event_instance_dict.pkl',
                       all_path + 'predict_diags_dict.pkl')
    train_size = int(x.shape[0] * train_perc)

    # shuffle the train set
    idx = np.random.permutation(x.shape[0])
    x_train = x[idx]
    y_train = y[idx]
    CsvUtility.write_array2csv(idx, '../data-repository/', 'random_idx.csv')
    return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:], idx


def reload_corpus(all_path='../data-repository/', train_perc=0.7, shuffle=False):
    x = CsvUtility.read_array_from_csv(all_path, 'feature_matrix.csv')
    y = CsvUtility.read_array_from_csv(all_path, 'result_matrix.csv')
    train_size = int(x.shape[0] * train_perc)
    # shuffle the train set
    if shuffle:
        idx = np.random.permutation(x.shape[0])
        CsvUtility.write_array2csv(idx, '../data-repository/', 'random_idx.csv')
    else:
        idx = CsvUtility.read_array_from_csv(all_path, 'random_idx.csv')
    x_train = x[idx]
    y_train = y[idx]
    return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:], idx

if __name__ == '__main__':
    training_x, training_y, testing_x, testing_y, idx = load_corpus()
    print training_x.shape
    print training_y.shape
    print testing_x.shape
    print testing_y.shape
    print len(idx)
