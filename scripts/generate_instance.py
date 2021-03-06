#!/usr/bin/env Python
# coding=utf-8

import cPickle as pickle
import pandas as pd
import numpy as np
from os import path
import os
import sys
import math
from random import randrange, shuffle
import json
import datetime
from collections import defaultdict
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
from utility.csv_utility import CsvUtility
from utility.nlp_utility import NLP_Utility

num_pred_diag = 80
balance_percent = 4 / 100

Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')


def get_first_index(value_list, value):
    if len(value_list) == 0:
        return 0
    for index, index_value in enumerate(value_list):
        if index_value >= value:
            return index
    return len(value_list)


def get_instance(time_before_diag = 90):
    print 'reading.....'
    all_events = CsvUtility.read_pickle(path.join(Path, 'data-repository/allevents.pkl'), 'r')
    print all_events.shape
    all_events.dropna(axis=0, how='any', inplace=True)
    print all_events.shape
    print 'changing the order......'
    all_events = all_events.ix[:, ['subject_id', 'charttime', 'event_type', 'event', 'hadm_id']]
    print all_events.dtypes
    # all_events = all_events.astype({'hadm_id': 'int64'})
    # print all_events.dtypes
    all_events['subject_id'] = all_events['subject_id'].astype('int64')
    for rr in all_events.ix[0, :]:
        print type(rr)
    print 'sorting ......'
    all_events.sort_values(by=['subject_id', 'charttime', 'event_type', 'event'],  inplace=True)
    print all_events[:10]
    rows = np.array(all_events, dtype=str)

    prev_time = None
    prev_subject = None
    # temp diagnoses in each time
    tem_diags = set()
    # temp event sequence in each time
    temp_event_seq = []
    # event sequence for each person
    event_seq = []
    # map the time for each person
    event_days = []
    # first time for each person
    base_time = None
    # all instance
    all_seq = []
    # whole set of events
    unique_events = set()
    # whole diagnoses count dict
    diag_count = defaultdict(lambda: 0)
    # count the length of instance
    seq_max = 0
    seq_min = 100000
    for i in rows[0]:
        print type(i)
    for i, row in enumerate(rows):
        # print i, row
        # if row[2] == "diagnosis":
        #     event = row[2][:1] + "_" + str(row[4])
        # else:
        #     event = row[2][:1] + "_" + str(row[3])
        event = row[2][:1] + "_" + str(row[3])

        # if type(row[1]) != str and math.isnan(row[1]):
        #     print 'delete nan:', row
        #     continue
        if prev_time is None or prev_subject is None:
            print 'first event'
            base_time = NLP_Utility.strtime2datetime(row[1])
        elif row[0] != prev_subject or NLP_Utility.strtime2datetime(row[1]) != prev_time:
            if len(tem_diags) > 0:
                # why exclude the diagnoses?
                # temp_event_seq = [x for x in temp_event_seq if x not in tem_diags]
                this_days = (prev_time - base_time).days
                find_days = this_days - time_before_diag if this_days >= time_before_diag else 0
                start_position = get_first_index(event_days, find_days)
                t_event_seq = []
                # for i_pos in range(start_position, len(event_days)):
                #     t_event_seq.append(event_seq[i_pos])
                    # unique_events.add(event_seq[i_pos])
                t_event_seq += event_seq[start_position:]
                # print len(event_seq[start_position:])
                # for test_event in event_seq[start_position:]:
                #     if test_event.startswith("p_"):
                #         print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                # for item in temp_event_seq:
                #     # t_event_seq.append(item)
                #     unique_events.add(item)
                all_seq.append([t_event_seq, list(tem_diags)])
                for iter_diag in tem_diags:
                    diag_count[iter_diag] = diag_count[iter_diag] + 1
                seq_max = seq_max if seq_max > len(t_event_seq) else len(t_event_seq)
                seq_min = seq_min if seq_min < len(t_event_seq) else len(t_event_seq)
            if row[0] != prev_subject:
                # print 'change patient ', row, ' pre: ', prev_subject, row[0]
                event_seq = []
                event_days = []
                base_time = NLP_Utility.strtime2datetime(row[1])
            else:
                # print 'change time ', row, ' pre: ', prev_time, row[1]
                event_seq += temp_event_seq
                # print prev_time
                # print base_time
                # print type((prev_time - base_time).days)
                event_days += [(prev_time - base_time).days] * len(temp_event_seq)
            tem_diags = set()
            temp_event_seq = []
        # print 'adding ....'
        temp_event_seq.append(event)
        prev_time = datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        prev_subject = row[0]
        if row[2] == "diagnosis":
            tem_diags.add(event)

        if i % 10000 == 0:
            print 'complete {0} of {1}'.format(i, len(rows))

    # Write down the vocalulary used and diagnoses that we want to predict
    predicted_diags = [y[0] for y in
                       sorted(diag_count.items(), key=lambda x: x[1], reverse=True)[:num_pred_diag]]
    print 'num of seq: ', len(all_seq)
    print all_seq[0]
    after_del_sequence = []
    for instance in all_seq:
        fil_diag = [diag for diag in instance[-1] if diag in predicted_diags]
        # if len(fil_diag) > 0:
        for item in instance[0]:
            unique_events.add(item)
        after_del_sequence.append(instance)
        after_del_sequence[-1][-1] = fil_diag
        for diag in fil_diag:
            unique_events.add(diag)
    print 'after limit the predict diagnoses, num of seq: ', len(after_del_sequence)
    print 'max/min of seq: ', seq_max, seq_min
    print 'number of unique items:', len(unique_events)
    CsvUtility.write2pickle(path.join(Path, 'data-repository/after_instance.pkl'), after_del_sequence, 'w')
    CsvUtility.write2pickle(path.join(Path, 'data-repository/event_instance_dict.pkl'), unique_events, 'w')
    CsvUtility.write2pickle(path.join(Path, 'data-repository/predict_diags_dict.pkl'), predicted_diags, 'w')
    print '************************************************************'
    #######################################################################################################

if __name__ == '__main__':
    # the forth step:
    get_instance()
    pass