
import numpy as np
import pandas as pd
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer

diagnoses_dict = {}
word_count = {}
stop_list = set(["of", "and", "by", "to", "or", "the", "in", "with"
                 , "not","classified", "for", "on", "from", "without"
                 , "as", "other", "than", "more", "at", "one", "all"
                 , "a", "its", "may", "after", "any", "d", "be", "into"
                 , "their", "which", "an"])

"""
# extract the dict of ICD9 codes and map it to its descriptions
with open("../data-repository/DC_3D11.txt",'r') as f:
    rows = f.readlines()
    i = 0
    for row in rows:
        words = row.strip().split("\t")
        if len(words)>1 and '.' not in words[0] and any(char.isdigit() for char in words[0]):
            assert words[0] not in diagnoses_dict
            wordlist = [re.sub("[^a-zA-Z]","",x.lower()) for x in words[1].split(' ')
                        if re.sub("[^a-zA-Z]","",x.lower()) not in stop_list]
            diagnoses_dict[words[0]] = wordlist
            for word in wordlist:
                word_count[word] = word_count[word] + 1 if word in word_count else 1
        i += 1
        sys.stdout.write('\rROW {0} of {1}...'.format(i, len(rows)))
    with open("../data-repository/diagnoses_dict.csv",'w') as w:
        for (key, value) in sorted(diagnoses_dict.items(), reverse= True):
            w.write(key + "," + " ".join(value) + "\n")
    with open("../data-repository/diagnoses_word_dict.csv",'w') as w:
        for (key, value) in sorted(word_count.items(), key = lambda s: s[1], reverse= True):
            w.write(key + "," + str(value) + "\n")
"""

data_diagnoses = pd.read_csv("../data-repository/diagnosis_counts.csv")
print 'data diagnoses size : ', data_diagnoses.shape[0]
icd_code = [x[:3] for x in data_diagnoses["ICD9_CODE"]]
data_diagnoses["ICD9_CODE"]=icd_code
data_diagnoses.set_index(["ICD9_CODE"], inplace = True)
diagnoses_dict = pd.read_csv("../data-repository/diagnoses_dict.csv", index_col=[0])
print 'diagnoses dict size :', diagnoses_dict.shape[0]
diagnoses_dict.index.name = 'ICD'
diagnoses_dict.columns = ['DESCRIPTION']
merge_diagnoses = diagnoses_dict.merge(data_diagnoses, how='inner', left_index=True, right_index=True)
merge_diagnoses.index.name = "ICD9CODE"
print 'after the merge, the diagnoses size : ', len(set(list(merge_diagnoses.index)))
# merge_diagnoses.to_csv("../data-repository/merge_diagnoses.csv")

# here can process the short title and long title from data, but here i use the standard ICD description
# ......
# merge_diagnoses["DESCRIPTION"].drop_duplicates().to_csv("../data-repository/merge_diagnoses_dict.csv")
count_vec = CountVectorizer()
re = count_vec.fit_transform(list(merge_diagnoses["DESCRIPTION"].drop_duplicates()))
print len(count_vec.get_feature_names())
print re.toarray().sum(axis=0)
count = re.toarray().sum(axis=0)
wordslist = count_vec.get_feature_names()
merge_diagnoses_word_count = {}
for i in range(len(wordslist)):
    merge_diagnoses_word_count[wordslist[i]] = count[i]
with open("../data-repository/merge_diagnoses_word_dict.csv", 'w') as w:
    for (key, value) in sorted(merge_diagnoses_word_count.items(), key=lambda s: s[1], reverse=True):
        w.write(key + "," + str(value) + "\n")









