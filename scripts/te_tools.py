
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

# with open('../new_literature/F_Adv_Healthc_Mater_2012_Sep_12_1(5)_640-645.txt', 'w') as f:
#     write_str = "hello."
#
#     f.write(write_str)
#Example #1: sum_primes.py

#!/usr/bin/python
# File: sum_primes.py
# Author: VItalii Vanovschi
# Desc: This program demonstrates parallel computations with pp module
# It calculates the sum of prime numbers below a given integer in parallel
# Parallel Python Software: http://www.parallelpython.com

import math, sys, time
import pp

def isprime(n):
    """Returns True if n is prime and False otherwise"""
    if not isinstance(n, int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i += 1
    return True

def sum_primes(n):
    """Calculates sum of all primes below given integer n"""
    return sum([x for x in xrange(2,n) if isprime(x)])

print """Usage: python sum_primes.py [ncpus]
    [ncpus] - the number of workers to run in parallel, 
    if omitted it will be set to the number of processors in the system
"""

# tuple of all parallel python servers to connect with
ppservers = ()
#ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print "Starting pp with", job_server.get_ncpus(), "workers"

# Submit a job of calulating sum_primes(100) for execution.
# sum_primes - the function
# (100,) - tuple with arguments for sum_primes
# (isprime,) - tuple with functions on which function sum_primes depends
# ("math",) - tuple with module names which must be imported before sum_primes execution
# Execution starts as soon as one of the workers will become available
job1 = job_server.submit(sum_primes, (100,), (isprime,), ("math",))

# Retrieves the result calculated by job1
# The value of job1() is the same as sum_primes(100)
# If the job has not been finished yet, execution will wait here until result is available
result = job1()

print "Sum of primes below 100 is", result

start_time = time.time()

# The following submits 8 jobs and then retrieves the results
inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
jobs = [(input, job_server.submit(sum_primes,(input,), (isprime,), ("math",))) for input in inputs]
for input, job in jobs:
    print "Sum of primes below", input, "is", job()

print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()

# Parallel Python Software: http://www.parallelpython.com