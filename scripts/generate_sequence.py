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

Path = '/Users/yangkai/Desktop/daily_work/MIMIC-III/data'
uniq_p_feat = ["gender", "age", "white", "asian", "hispanic", "black", "multi", "portuguese",
               "american", "mideast", "hawaiian", "other"]

# seq_path = '../../Data/mimic_seq/'
# balanced_seq_path = '../../Data/mimic_balanced/'
num_pred_diag = 80
balance_percent = 4 / 100

def set_p_features():

    admission_df = pd.read_csv(os.path.join(Path, 'ADMISSIONS.csv'))[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY']]
    print admission_df[:5]
    print admission_df.shape
    patients_df = pd.read_csv(os.path.join(Path, 'PATIENTS.csv'))[['SUBJECT_ID', 'GENDER', 'DOB']]
    print patients_df[:5]
    print patients_df.shape
    merge_df = pd.merge(admission_df, patients_df, how='inner', on='SUBJECT_ID')
    print merge_df[:10]
    print merge_df.shape

    # cur.execute("""SELECT dob, admittime, gender, ethnicity from admissions join patients
    #             on admissions.subject_id = patients.subject_id
    #             where hadm_id = %(hadm_id)s """ % {'hadm_id': str(hadm_id)})
    # subject_info = cur.fetchall()

    merge_df.sort_values(by=['HADM_ID'], inplace=True)
    print merge_df.dtypes
    merge_feature = np.array(merge_df)
    feature_map = {}
    pre_hadm = None
    for line in merge_feature:
        if line[1] != pre_hadm:
            feats = {}
            for k in uniq_p_feat:
                feats[k] = 0
            feats["gender"] = int(line[4] == "M")
            num_years = (NLP_Utility.strtime2datetime(line[2]) - NLP_Utility.strtime2datetime(line[5])).days / 365.25
            feats["age"] = num_years

            r = line[3]
            if "WHITE" in r:
                feats["white"] = 1
            elif "ASIAN" in r:
                feats["asian"] = 1
            elif "HISPANIC" in r:
                feats["hispanic"] = 1
            elif "BLACK" in r:
                feats["black"] = 1
            elif "MULTI" in r:
                feats["multi"] = 1
            elif "PORTUGUESE" in r:
                feats["portuguese"] = 1
            elif "AMERICAN INDIAN" in r:
                feats["american"] = 1
            elif "MIDDLE EASTERN" in r:
                feats["mideast"] = 1
            elif "HAWAIIAN" in r or "CARIBBEAN" in r:
                feats["hawaiian"] = 1
            else:
                feats["other"] = 1
            feature_map[line[1]] = feats
            pre_hadm = line[1]
    return feature_map


def get_sequence():
    print 'reading.....'
    all_events = CsvUtility.read_pickle('../data-repository/allevents.pickle', 'r')
    print all_events.shape
    all_events.dropna(axis=0, how='any', subset=['subject_id', 'charttime', 'event', 'hadm_id'], inplace=True)
    print all_events.shape
    print 'changing the order......'
    all_events = all_events.ix[:, ['subject_id', 'charttime', 'event_type', 'event', 'icd9_3', 'hadm_id']]
    print all_events.dtypes
    all_events = all_events.astype({'hadm_id': 'int64'})
    print all_events.dtypes
    print 'sorting ......'
    all_events.sort_values(by=['subject_id', 'hadm_id', 'charttime', 'event_type', 'event'],  inplace=True)
    print all_events[:10]
    rows = np.array(all_events)

    prev_time = None
    prev_subject = None
    prev_hadm_id = None
    # temp diagnoses in each admission
    diags = set()
    # temp event sequence in each admission
    temp_event_seq = []
    event_seq = []
    # event sequence for each person
    all_seq = []
    # map the time to the events in all_seq
    all_days = []
    # whole set of events
    unique_events = set()
    # whole diagnoses count dict
    diag_count = defaultdict(lambda: 0)
    # get the static feature of a patient
    p_features = set_p_features()
    # count the length of sequence
    seq_len = 0
    seq_max = 0
    seq_min = 100000
    for i in rows[0]:
        print type(i)
    for i, row in enumerate(rows):
        # print i, row
        if row[2] == "diagnosis":
            event = row[2][:1] + "_" + str(row[4])
            if not row[2].startswith("E"):
                diag_count[event] += 1
        else:
            event = row[2][:1] + "_" + str(row[3])

        if row[0] is None or row[1] is None or row[5] is None:
            print 'delete None:', row
            continue
        elif type(row[1]) != str and math.isnan(row[1]):
            print 'delete nan:', row
            continue

        elif prev_time is None or prev_subject is None:
            print 'first event'
            pass

        elif (row[0] != prev_subject) or (NLP_Utility.strtime2datetime(row[1]) > prev_time + datetime.timedelta(365)):
            print 'change sequence', row, ' pre: ', prev_subject, prev_time
            if len(diags) > 0 and len(event_seq) > 4:
                # pre, suf = calculate_window(event_seq + temp_event_seq, all_days)
                # all_seq.append([p_features, event_seq, temp_event_seq, diags, pre, suf])
                temp_event_seq = [x for x in temp_event_seq if x not in diags]
                for item in event_seq:
                    unique_events.add(item)
                for item in temp_event_seq:
                    unique_events.add(item)
                all_days.append(len(temp_event_seq))
                all_seq.append([p_features[prev_hadm_id], event_seq, temp_event_seq, all_days, diags])
                print '!!!__!!!', prev_subject
                print len(event_seq)+len(temp_event_seq), len(all_days), sum(all_days)
                seq_len += len(all_days)
                seq_max = seq_max if seq_max > len(all_days) else len(all_days)
                seq_min = seq_min if seq_min < len(all_days) else len(all_days)
            diags = set()
            event_seq = []
            temp_event_seq = []
            all_days = []
        elif prev_hadm_id != row[5]:
            print 'change temp sequence:', row, ' prev: ', prev_hadm_id
            all_days.append(len(temp_event_seq))
            event_seq += temp_event_seq
            temp_event_seq = []
            diags = set()
        elif NLP_Utility.strtime2datetime(row[1]) != prev_time :
            # print 'just change time: ', prev_time, rows[1]
            all_days.append(len(temp_event_seq))
            event_seq += temp_event_seq
            temp_event_seq = []

        # print 'adding ....'
        temp_event_seq.append(event)

        prev_time = datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        prev_subject = row[0]
        prev_hadm_id = row[5]

        if row[2] == "diagnosis":
            diags.add(event)

        if i%10000 == 0:
            print 'complete {0} of {1}'.format(i, len(rows))

    # Write down the vocalulary used and diagnoses that we want to predict
    predicted_diags = [y[0] for y in
                       sorted(diag_count.items(), key=lambda x: x[1], reverse=True)[:num_pred_diag]]

    # uniq = open('../data-repository/vocab', 'w')
    # uniq.write(' '.join(unique_events) + '\n')
    # uniq.write(' '.join(predicted_diags))
    # uniq.close()
    print len(all_seq)
    print all_seq[0]
    after_del_sequence =[]
    for instance in all_seq:
        fil_diag = [diag for diag in instance[-1] if diag in predicted_diags]
        if len(fil_diag)>0:
            after_del_sequence.append(instance)
            after_del_sequence[-1][-1] = fil_diag
    print 'num of seq: ', len(after_del_sequence)
    print 'max/min of seq: ', seq_max, seq_min
    print 'mean of seq: ', seq_len/len(after_del_sequence)
    CsvUtility.write2pickle('../data-repository/after_sequence.pickle', after_del_sequence, 'w')
    CsvUtility.write2pickle('../data-repository/event_dict.pickle', unique_events, 'w')


    print '************************************************************'
    #######################################################################################################

    def get_diag_sequence():
        pass

if __name__ == '__main__':
    # set_p_features()
    get_sequence()
