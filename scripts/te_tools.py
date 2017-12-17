
import pandas as pd
import numpy as np
from utility.csv_utility import CsvUtility
from utility.nlp_utility import NLP_Utility
import datetime
from torch.autograd import Variable
import torch
from scipy.special import psi
from gensim import matutils

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
# tmp = "34-Mg"
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

# with open('../new_literature/F_Adv_Healthc_Mater_2012_Sep_12_1(5)_640-645.txt', 'w') as f:
#     write_str = "hello."
#
#     f.write(write_str)
#Example #1: sum_primes.py
'''
gamma_data = np.array(pd.read_csv('../data-repository/gamma_from_LDA.csv',
                         index_col=None, header=None)).T
print gamma_data.shape
print gamma_data[:20]

gamma_id2word = {}
with open('../data-repository/available_word_in_literature.csv') as file:
    line_num = file.readline()
    # print line_num
    lines_contend = file.readlines()
    for line_n in lines_contend:
        line = line_n.split("\t")
        # print line
        if len(line) > 1:
            gamma_id2word[int(line[0])] = line[1]
print len(gamma_id2word)
id_list = gamma_id2word.keys()
print np.array(id_list).max()


feature_index = pd.read_csv('../data-repository/feature2index.csv',
                            header=None, index_col=None)
print feature_index.shape
print feature_index[:5]
f_i = np.array(feature_index)
print f_i.shape, f_i[:, 1].max()

feature_word2id = {}
# np.zeros((feature_index.shape[0], gamma_data.shape[1]))
for i in range(f_i.shape[0]):
    feature_word2id[f_i[i][0]] = int(f_i[i][1])
print len(feature_word2id)

change_index_result = np.zeros((feature_index.shape[0],
                                gamma_data.shape[1]))
for i in range(gamma_data.shape[0]):
    new_index = feature_word2id[gamma_id2word[i]]
    for j in range(gamma_data.shape[1]):
        change_index_result[new_index][j] += gamma_data[i][j]
    if i%1000 == 0:
        print i, 'line'
print change_index_result[:5]
print change_index_result.shape
''''''
import torch
from torch.autograd import Variable
gamma_test = -1*Variable(torch.from_numpy(np.random.rand(3, 6)))
para_test = Variable(torch.from_numpy(np.random.rand(6, 4)))
print gamma_test
print gamma_test.abs()
print (gamma_test.abs().sum(dim=1).view(-1, 1))
print (gamma_test.abs()) / (gamma_test.abs().sum(dim=1).view(-1, 1))


print para_test
mum = gamma_test.abs().mm(para_test)
print mum
print 'max: '
max = mum.max(dim=1)[0]
print max
sum = max.sum()
print sum
re = Variable(torch.DoubleTensor([1.0]))/sum
print re
'''
a = Variable(torch.FloatTensor([1.1]))
print a
print psi([1.0])
# print psi(a)
print matutils.dirichlet_expectation(np.array([1.0]))
print matutils.dirichlet_expectation(a.data)

print str('"yang"').replace('"', '')