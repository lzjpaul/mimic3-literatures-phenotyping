#!/usr/bin/env Python
# coding=utf-8
import os
import re
import sys
import pandas as pd
import numpy as np
import argparse
from time import clock
import cPickle as pickle

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.csv_utility import CsvUtility
from utility.directory_utility import Directory
from utility.nlp_utility import NLP_Utility


Path = '/Users/yangkai/Desktop/daily_work/MIMIC-III/data'

def icd_diagnoses_over(filename, over_num):
    diagnoses_df = pd.read_csv(os.path.join(Path, filename))[['HADM_ID', 'ICD9_CODE']]
    print diagnoses_df[:5]
    print diagnoses_df.shape
    diagnoses_df.drop_duplicates(inplace=True)
    print diagnoses_df.shape
    diag_count = diagnoses_df['ICD9_CODE'].value_counts()
    diag_df = pd.DataFrame(diag_count[diag_count > over_num])
    diag_df.columns = ['COUNT']
    diag_df.index.name = 'ICD9_CODE'
    print diag_df[:5]
    print 'size:', diag_df.shape
    CsvUtility.write2pickle('../data-repository/icd_diagnoses_over.pickle', diag_df, 'w')


def subject_admission_over(filename, over_num):
    admission_df = pd.read_csv(os.path.join(Path, filename))
    # print admission_df[:5]
    # admission_df.filter(items=['SUBJECT_ID'], like=)
    sub_vc = admission_df['SUBJECT_ID'].value_counts()
    sub_df = pd.DataFrame(sub_vc[sub_vc > over_num])
    sub_df.columns = ['COUNT']
    sub_df.index.name = 'SUBJECT_ID'
    print sub_df[:5]
    print 'size: ', sub_df.shape
    with open('../data-repository/subject_admission_over.pickle', 'w') as f:
        pickle.dump(sub_df, f)
        f.close()


def get_all_diagnoses_event():

    diagnoses_df = pd.read_csv(os.path.join(Path, 'DIAGNOSES_ICD.csv'))
    # print diagnoses_df[:5]
    admission_df = pd.read_csv(os.path.join(Path, 'ADMISSIONS.csv'))
    # print admission_df[:5]
    diagnoses_event = pd.merge(diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']],
                               admission_df[['HADM_ID', 'DISCHTIME', 'DIAGNOSIS']], 'left', on='HADM_ID')
    diagnoses_event['DIAGNOSIS'] = ['diagnosis'] * diagnoses_event.shape[0]
    # print diagnoses_event[:10]
    # print diagnoses_event.shape
    icd_df = CsvUtility.read_pickle('../data-repository/icd_diagnoses_over.pickle', 'r')
    sub_df = CsvUtility.read_pickle('../data-repository/subject_admission_over.pickle', 'r')
    diagnoses_set = list(pd.read_csv('../data-repository/merge_diagnoses_dict.csv', header=None).index)
    diagnoses_event = diagnoses_event[diagnoses_event['SUBJECT_ID'].isin(list(sub_df.index)) &
                                                       diagnoses_event['ICD9_CODE'].isin(list(icd_df.index))]
    ######################################
    print 'additional process'
    np_diagnoses_event = np.array(diagnoses_event)
    new_diagnoses_event = []
    diagnoses_set = set(list(pd.read_csv('../data-repository/merge_diagnoses_dict.csv', header=None)[0]))
    for i in range(len(np_diagnoses_event)):
        if np_diagnoses_event[i][2] != np.NaN and len(np_diagnoses_event[i][2]) >= 3 and np_diagnoses_event[i][2][:3] in diagnoses_set:
            new_line = []
            new_line.extend(np_diagnoses_event[i])
            new_line.append(np_diagnoses_event[i][2][:3])
            if re.match('^V.*', np_diagnoses_event[i][2]):
                new_line[4] = 'condition'
            if re.match('^7[89]\d.*', np_diagnoses_event[i][2]):
                new_line[4] = 'symptom'
            new_diagnoses_event.append(new_line)
        if i % 10000 == 0:
            print i
    new_columns = list(diagnoses_event.columns)
    new_columns.append('icd9_3')
    print new_columns
    print new_diagnoses_event[:5]
    diagnoses_event = pd.DataFrame(new_diagnoses_event)
    diagnoses_event.columns = new_columns

    ######################################

    print diagnoses_event[:10]
    print diagnoses_event.shape
    print len(set(list(diagnoses_event['ICD9_CODE']))), len(set(list(diagnoses_event['icd9_3'])))
    return diagnoses_event

def get_lab_item_over(file_name, over_num):
    labevent_df = pd.read_csv(os.path.join(Path, file_name))[['HADM_ID', 'ITEMID', 'FLAG']]
    print labevent_df[:5]
    print labevent_df.shape
    labevent_df = labevent_df[labevent_df['FLAG'] == 'abnormal']
    print labevent_df.shape
    labevent_df.drop_duplicates(inplace=True)
    print labevent_df.shape
    item_count = labevent_df['ITEMID'].value_counts()
    item_df = pd.DataFrame(item_count[item_count > over_num])
    item_df.columns = ['COUNT']
    item_df.index.name = 'ITEMID'
    print item_df[:5]
    print 'size:', item_df.shape
    CsvUtility.write2pickle('../data-repository/lab_item_over.pickle', item_df, 'w')


def get_lab_event():
    labevent_df = pd.read_csv(os.path.join(Path, 'LABEVENTS.csv'), dtype={'HADM_ID': str})[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'FLAG']]
    labevent_df = labevent_df[labevent_df['FLAG'] == 'abnormal']
    labevent_df['FLAG']=['labevent'] * labevent_df.shape[0]
    # labevent_df['SUBJECT_ID'] = labevent_df['SUBJECT_ID'].astype('str')
    # labevent_df['HADM_ID'] = labevent_df['HADM_ID'].astype('str')
    print labevent_df[-5:]
    print labevent_df.shape
    print labevent_df.dtypes
    sub_df = CsvUtility.read_pickle('../data-repository/subject_admission_over.pickle', 'r')
    item_df = CsvUtility.read_pickle('../data-repository/lab_item_over.pickle', 'r')
    labevent_df = labevent_df[labevent_df['SUBJECT_ID'].isin(list(sub_df.index)) &
                              labevent_df['ITEMID'].isin(list(item_df.index))]
    labevent_df['icd9_3'] = [''] * labevent_df.shape[0]
    print labevent_df.shape
    print len(set(list(labevent_df['ITEMID'])))
    return labevent_df

def get_drug_over(file_name, over_num):
    drug_df = pd.read_csv(os.path.join(Path, file_name))[['HADM_ID', 'FORMULARY_DRUG_CD']]
    print drug_df[:5]
    print drug_df.shape
    drug_df.drop_duplicates(inplace=True)
    print drug_df.shape
    drug_count = drug_df['FORMULARY_DRUG_CD'].value_counts()
    drug_df = pd.DataFrame(drug_count[drug_count > over_num])
    drug_df.columns = ['COUNT']
    drug_df.index.name = 'FORMULARY_DRUG_CD'
    print drug_df[:5]
    print 'size:', drug_df.shape
    CsvUtility.write2pickle('../data-repository/prescription_drug_over.pickle', drug_df, 'w')

def get_medication_event():
    medication_df = pd.read_csv(os.path.join(Path, 'PRESCRIPTIONS.csv'))[['SUBJECT_ID', 'HADM_ID', 'STARTDATE', 'DRUG_TYPE', 'FORMULARY_DRUG_CD']]

    # print medication_df[:5]
    medication_df['DRUG_TYPE'] = ['prescription'] * medication_df.shape[0]
    # print medication_df[:5]
    # print medication_df.shape
    sub_df = CsvUtility.read_pickle('../data-repository/subject_admission_over.pickle', 'r')
    drug_df = CsvUtility.read_pickle('../data-repository/prescription_drug_over.pickle', 'r')
    medication_df = medication_df[medication_df['SUBJECT_ID'].isin(list(sub_df.index)) &
                                  medication_df['FORMULARY_DRUG_CD'].isin(list(drug_df.index))]
    medication_df ['icd9_3'] = [''] * medication_df.shape[0]
    print medication_df.shape
    print len(set(list(medication_df['FORMULARY_DRUG_CD'])))
    return medication_df

def get_events_together():
    columns_name = ['hadm_id', 'subject_id', 'charttime', 'event_type', 'event', 'icd9_3']

    diag_columns = ['HADM_ID', 'SUBJECT_ID', 'DISCHTIME', 'DIAGNOSIS', 'ICD9_CODE', 'icd9_3']
    diag_events = get_all_diagnoses_event()
    diag_events = diag_events.ix[:, diag_columns]
    diag_events.columns = columns_name
    print diag_events[:5]

    lab_events = get_lab_event()
    lab_columns = ['HADM_ID', 'SUBJECT_ID', 'CHARTTIME', 'FLAG', 'ITEMID', 'icd9_3']
    lab_events = lab_events.ix[:, lab_columns]
    lab_events.columns = columns_name
    print lab_events[:5]

    medic_events = get_medication_event()
    medic_columns = ['HADM_ID', 'SUBJECT_ID', 'STARTDATE', 'DRUG_TYPE', 'FORMULARY_DRUG_CD', 'icd9_3']
    medic_events = medic_events.ix[:, medic_columns]
    medic_events.columns = columns_name
    print medic_events[:5]

    all_events = pd.concat([diag_events, lab_events, medic_events], ignore_index=True)
    print all_events[:5]
    print all_events[-5:]
    print all_events.shape

    CsvUtility.write2pickle('../data-repository/allevents.pickle', all_events, 'w')

    # all_events = CsvUtility.read_pickle('../data-repository/allevents.pickle', 'r')
    # print all_events.shape
    # all_events = all_events[all_events['event'] != '']
    # print all_events.shape
    # all_events.dropna(axis=0, how='any', subset=['event'], inplace=True)
    # print all_events.shape

def filter_all_event():
    all_events_df = CsvUtility.read_pickle('../data-repository/allevents.pickle', 'r')
    all_events_df['icd9_3'] = ''
    print all_events_df[:5]
    print all_events_df.shape
    # diagnoses_events = all_events_df[all_events_df['event_type'] == 'diagnosis']
    # print diagnoses_events[:5]
    # print diagnoses_events.shape
    # diagnoses_set = set(list(pd.read_csv('../data-repository/merge_diagnoses_dict.csv', header=None).index))
    # print len(diagnoses_set)
    # i=0
    # for index_iter in diagnoses_events.index:
    #     icd_code = diagnoses_events.ix[index_iter, 'event']
    #     assert len(icd_code) >= 3
    #     if len(icd_code) >= 3:
    #         if icd_code[:3] in diagnoses_set:
    #             all_events_df.ix[index_iter, 'icd9_3'] = all_events_df.ix[index_iter, 'event'][:3]
    #         else:
    #             all_events_df.drop(index_iter, axis=0, inplace=True)
    #     sys.stdout.write('\rROW {0} of {1}...'.format(i, diagnoses_events.shape[0]))
    #     i += 1
    # all_events_df.index = np.array(range(all_events_df.shape[0]))
    print all_events_df[:5]
    print all_events_df.shape
    CsvUtility.write2pickle('../data-repository/all_events_icd9.pickle', all_events_df, 'w')







if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Extract EHR data from MIMICIII dataset.")
    # parser.add_argument('doc_origin_path', type=str,
    #                     help='Original dataset directory may containing several folders.')
    # parser.add_argument('output_path', type=unicode, help='Directory where output data should be written.')
    # parser.add_argument('entity_file', type=unicode, help='Name of appending part of csv file')
    # args, _ = parser.parse_known_args()
    # try:
    #     os.makedirs(args.output_path)
    # except Exception:
    #     pass
    # subject_admission_over('ADMISSIONS.csv', 1)
    # icd_diagnoses_over('DIAGNOSES_ICD.csv', 5)
    # get_all_diagnoses_event()
    # get_lab_item_over('LABEVENTS.csv', 50)
    # get_lab_event()
    # get_drug_over('PRESCRIPTIONS.csv', 50)
    # get_medication_event()
    get_events_together()
    # filter_all_event()
    print '******************************************************************************'
#test code

#python select_relate_literature.py '../data-repository/BMC_Musuloskelet_Disord' '../data-repository' 'merge_diagnoses_word_dict.csv'
