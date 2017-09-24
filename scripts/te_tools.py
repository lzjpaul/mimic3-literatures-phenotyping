
import pandas as pd
import numpy as np
from utility.csv_utility import CsvUtility
from utility.nlp_utility import NLP_Utility
import datetime

# frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=['two', 'one'], columns=['d', 'a', 'b', 'c'])
#
# frame = frame.append(frame)
# frame.ix[0, 1] = 10
# frame.ix[1, 1] = 2
# print frame
#
# print frame.sort_values(by=['d', 'a'], ascending=False, inplace=True)
# print np.array(frame)
# # print frame.sort_index(axis=1)
# # print
# print frame.dtypes
# frame = frame.astype({'a':str})
# print frame.dtypes
# print type(frame.ix[2, 1])
# print 'reading.....'
# all_events = CsvUtility.read_pickle('../data-repository/allevents.pkl', 'r')
# print all_events.shape
# all_events.dropna(axis=0, how='any', subset=['subject_id', 'charttime', 'event', 'hadm_id'], inplace=True)
# print all_events.shape
# print 'changing the order......'
# all_events = all_events.ix[:, ['subject_id', 'charttime', 'event_type', 'event', 'icd9_3', 'hadm_id']]
# print 'sorting ......'
# all_events.sort_values(by=['subject_id', 'hadm_id', 'charttime', 'event_type', 'event'], ascending=False, inplace=True)
# print all_events[:10]
# print all_events.dtypes
# all_events = all_events.astype({'hadm_id': 'int64'})
# print all_events.dtypes
# all_events.to_csv('../data-repository/ all_temp.csv')
# print (NLP_Utility.strtime2datetime('2119-02-03 01:35:00') - NLP_Utility.strtime2datetime('2119-02-02 01:35:00')).days
# a=[1,1,1,1,3,3,4,4]
# start = a.index(0) if a.__contains__(0) else 0
# print range(start, len(a))
# print 1/(3*1.0)
# print np.ceil(1/(3*1.0))
# a = [1,2,3,4,5]
# a.extend([44])
# print a
# a = "alteplase 1mg/flush volume ( dialysis/pheresis catheters )"
# for word_tmp in a.split(" "):
#     print word_tmp
#     tmp = word_tmp.lower()
#     if len(tmp) > 0 and any(char.isalpha() for char in tmp):
#         print tmp
# print a.isalpha()
# print a
# print a[:-1]
tmp = "34-Mg"
# print tmp.endswith("mg")
# print tmp[:-2].isdigit()
# if tmp.endswith("mg") and len(tmp) > 2 and tmp[:-2].isdigit():
#     print "OK"
# import re
# print re.sub("[^a-zA-Z-]", "", tmp.lower())
# rr = 4
# print rr in [1,2,3,4]
# import pandas as pd
# import numpy as np
#
# a = pd.DataFrame(np.random.randint(0, 7, size=(4, 5)))
# print a
# print type(np.array(a))
# print type(a.values)
# for i in range(8, 8):
#     print i
# a = [1,2,3,4,5]
# print a + a
from  utility.csv_utility import CsvUtility
from utility.directory_utility import Directory

# print 'reading.....'
# all_events = CsvUtility.read_pickle('../data-repository/allevents.pkl', 'r')
# print all_events.shape
# all_events.dropna(axis=0, how='any', inplace=True)
# print all_events.shape
# print 'changing the order......'
# all_events = all_events.ix[:, ['subject_id', 'charttime', 'event_type', 'event', 'hadm_id']]
# print all_events.dtypes
# all_events['subject_id'] = all_events['subject_id'].astype('int64')
#
# for rr in all_events.ix[0, :]:
#     print type(rr)
#
# # all_events = all_events.astype({'hadm_id': 'int64'})
# # print all_events.dtypes
# print 'sorting ......'
# all_events.sort_values(by=['subject_id', 'charttime', 'event_type', 'event'], inplace=True)
# print all_events[:10]
# all_events.to_csv('../data-repository/test_all_events.csv', index=None, header=None)

# a = pd.DataFrame([['1', '2', '2', '5'],
#                  ['11', '22', '2', '5'],
#                  ['03', '12', '3', '5'],
#                  ['1', '9', '11', '5'],
#                  ['111', '2', '0', '5']])
# print a
# a.sort_values(by=[0,1,2], inplace=True)
# print a
import os

with open('../new_literature/F_Adv_Healthc_Mater_2012_Sep_12_1(5)_640-645.txt', 'w') as f:
    write_str = "hello."

    f.write(write_str)
