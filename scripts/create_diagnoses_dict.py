
import numpy as np
import pandas as pd
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer

from utility.csv_utility import CsvUtility

diagnoses_dict = {}
word_count = {}
stop_list = {"of", "and", "by", "to", "or", "the", "in", "with"
                 , "not","classified", "for", "on", "from", "without"
                 , "as", "other", "than", "more", "at", "one", "all"
                 , "a", "its", "may", "after", "any", "d", "be", "into"
                 , "their", "which", "an", "*nf", "nf*", "but"}


# extract the dict of ICD9 codes and map it to its descriptions from the standard file DC_3D11.txt
# with open("../data-repository/DC_3D11.txt", 'r') as f:
#     rows = f.readlines()
#     i = 0
#     for row in rows:
#         words = row.strip().split("\t")
#         if len(words) > 1 and '.' not in words[0] and any(char.isdigit() for char in words[0]):
#             assert words[0] not in diagnoses_dict
#             wordlist = [re.sub("[^a-zA-Z-]", "", x.lower()) for x in words[1].split(' ')
#                         if re.sub("[^a-zA-Z-]", "", x.lower()) not in stop_list]
#             diagnoses_dict[words[0].strip()] = wordlist
#             for word in wordlist:
#                 word_count[word] = word_count[word] + 1 if word in word_count else 1
#         i += 1
#         sys.stdout.write('\rROW {0} of {1}...'.format(i, len(rows)))
#     with open("../data-repository/diagnoses_dict.csv", 'w') as w:
#         for (key, value) in sorted(diagnoses_dict.items(), reverse=True):
#             w.write(key + "," + " ".join(value) + "\n")
#     with open("../data-repository/diagnoses_word_dict.csv", 'w') as w:
#         for (key, value) in sorted(word_count.items(), key=lambda s: s[1], reverse=True):
#             w.write(key + "," + str(value) + "\n")

# merge the ICD code with the ICD code from the MIMIC data
# data_diagnoses = pd.read_csv("../data-repository/diagnosis_counts.csv")
# print 'data diagnoses size : ', data_diagnoses.shape[0]
# icd_code = [x[:4] if x[0] == 'E' else x[:3] for x in data_diagnoses["ICD9_CODE"]]
# # print icd_code
# data_diagnoses["ICD9_CODE"] = icd_code
# data_diagnoses.set_index(["ICD9_CODE"], inplace=True)
# diagnoses_dict = pd.read_csv("../data-repository/diagnoses_dict.csv", index_col=[0])
# print 'diagnoses dict size :', diagnoses_dict.shape[0]
# diagnoses_dict.index.name = 'ICD'
# diagnoses_dict.columns = ['DESCRIPTION']
# merge_diagnoses = diagnoses_dict.merge(data_diagnoses, how='inner', left_index=True, right_index=True)
# merge_diagnoses.index.name = "ICD9CODE"
# print 'after the merge, the diagnoses size : ', len(set(list(merge_diagnoses.index)))
# merge_diagnoses.to_csv("../data-repository/merge_diagnoses.csv")

# here can process the short title and long title from data, but here i use the standard ICD description
# ......
# merge_diagnoses["DESCRIPTION"].drop_duplicates().to_csv("../data-repository/merge_diagnoses_dict.csv")
# count_vec = CountVectorizer()
# re = count_vec.fit_transform(list(merge_diagnoses["DESCRIPTION"].drop_duplicates()))
# print len(count_vec.get_feature_names())
# print re.toarray().sum(axis=0)
# count = re.toarray().sum(axis=0)
# wordslist = count_vec.get_feature_names()
# merge_diagnoses_word_count = {}
# for i in range(len(wordslist)):
#     merge_diagnoses_word_count[wordslist[i]] = count[i]
# with open("../data-repository/merge_diagnoses_word_dict.csv", 'w') as w:
#     for (key, value) in sorted(merge_diagnoses_word_count.items(), key=lambda s: s[1], reverse=True):
#         w.write(key + "," + str(value) + "\n")

# ====================================================================================================
# 2017/9/10 Peking University
# Adding the information of medication and labtest into the model
# extract the dict of prescription and map it to its description

import argparse
# input the parameters by the args, original path of MIMIC III is "E:\medical data\MIMICIII_data", and demo data is \
# "E:\medical data\MIMICIII_demo_data

# parser = argparse.ArgumentParser(description="Extract the dict of map (prescription/labtest 2 description).")
# parser.add_argument('mimic_original_path',  type=str, help='Original path of MIMIC III data.')
# parser.add_argument('output_path', type=str, help='Directory where output data should be written.')
# args,_ = parser.parse_known_args()
# try:
#     os.makedirs(args.output_path)
# except Exception:
#     pass


def remove_bracket_from_str(str):
    if  len(str) == 0:
        return str
    bracket_flag = False
    bracket_left_index = -1
    new_str = ""
    for i in range(len(str)):
        if str[i] == '(':
            bracket_flag = True
            bracket_left_index = i
        elif str[i] == ')':
            bracket_flag = False
        else:
            if bracket_flag == False:
                new_str += str[i]
    if bracket_flag:
        new_str += str[bracket_left_index:]
    return new_str.strip()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass


def remove_quotation_from_str(str):
    if str[0] == '\"' and str[-1] == '\"':
        str_re = str.replace("\"", "")
        str_tmp = str_re.split(",")
        if len(str_tmp) == 2:
            str_re = str_tmp[0]
        return str_re
    return str


# original_path = 'E:\medical data\MIMICIII_data'

# prescription_df = pd.read_csv(os.path.join(original_path, 'PRESCRIPTIONS.csv'), dtype=str)
# print prescription_df.shape
# print prescription_df[:5]
# print prescription_df.dtypes
# print prescription_df.describe()
# prescription_dict = prescription_df[['FORMULARY_DRUG_CD', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC']]
# print prescription_dict.shape
# prescription_dict = prescription_dict.dropna()
# print prescription_dict.shape
# prescription_dict = prescription_dict.drop_duplicates()
# print prescription_dict.shape
#
# stop_char = ['(', ')', '/', '/"', '-']
# stop_str = set(["*nf*", "a", "b","of", "and", "by", "to", "or", "the", "in", "with"
#                  , "not", "classified", "for", "on", "from", "without"
#                  , "as", "other", "than", "more", "at", "one", "all"
#                  , "its", "may", "after", "any", "d", "be", "into"
#                  , "their", "which", "an", "ec", "c", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"
#                  , "q", "i", "s", "t", "u", "v", "w", "x", "y", "z", "vs.", "mg", "extended-release", ""])
# revert_prescrip_dict = {}
# prescrip_list = prescription_dict.values
# print prescrip_list[:5]
# for i in range(len(prescrip_list)):
#     word_list_tmp = []
#     prescrip_str = remove_bracket_from_str(prescrip_list[i][1])
#     for stop_c in stop_char:
#         prescrip_str = prescrip_str.replace(stop_c, ' ').strip()
#     for word_tmp in prescrip_str.split(" "):
#         tmp = word_tmp.lower()
#         if len(tmp) > 0 and any(char.isalpha() for char in tmp):
#             if tmp.endswith("mg") and len(tmp) > 2 and is_number(tmp[:-2]):
#                 pass
#             elif tmp not in stop_str:
#                 word_list_tmp.append(tmp.strip())
#     words = " ".join(word_list_tmp).strip()
#     if len(words) > 0 and words not in revert_prescrip_dict:
#         revert_prescrip_dict[words] = prescrip_list[i][0]
#
#     word_list_tmp = []
#     prescrip_str = remove_bracket_from_str(prescrip_list[i][2])
#     for stop_c in stop_char:
#         prescrip_str = prescrip_str.replace(stop_c, ' ').strip()
#     for word_tmp in prescrip_str.split(" "):
#         tmp = word_tmp.lower()
#         if len(tmp) > 0 and any(char.isalpha() for char in tmp):
#             if tmp.endswith("mg") and len(tmp) > 2 and is_number(tmp[:-2]):
#                 pass
#             elif tmp not in stop_str:
#                 word_list_tmp.append(tmp.strip())
#     words = " ".join(word_list_tmp).strip()
#     if len(words) > 0 and words not in revert_prescrip_dict:
#         revert_prescrip_dict[words] = prescrip_list[i][0]
# print revert_prescrip_dict
# print len(revert_prescrip_dict)

# print prescription_dict[:5]
# prescription_dict.to_csv("../data-repository/prescription_dict.csv", index=None)


# CsvUtility.write_text2csv(dict(revert_prescrip_dict), "../data-repository", 'revert_prescription_dict.csv')

# labtest revert dict, same to prescription
# labtest_df = pd.read_csv(os.path.join(original_path, 'D_LABITEMS.csv'), dtype=str)
# print labtest_df.shape
# print labtest_df[:5]
# print labtest_df.dtypes
# print labtest_df.describe()
# labtest_dict = labtest_df[['ITEMID', 'LABEL']]
# print labtest_dict.shape
# labtest_dict = labtest_dict.dropna()
# print labtest_dict.shape
# labtest_dict = labtest_dict.drop_duplicates()
# print labtest_dict.shape
# print labtest_dict[:5]
# labtest_dict.to_csv("../data-repository/labtest_dict.csv", index=None)
#
# labtest_list = labtest_dict.values
# revert_labtest_dict = {}
# for i in range(len(labtest_list)):
#     temp_str = remove_bracket_from_str(labtest_list[i][1])
#     temp_str = remove_quotation_from_str(temp_str)
#     temp_str = temp_str.replace(",", " ").strip().lower()
#     revert_labtest_dict[temp_str] = labtest_list[i][0]
#
# CsvUtility.write_text2csv(dict(revert_labtest_dict), "../data-repository", "revert_labtest_dict.csv")

# why limit the ICD code to 3, and that will make the number of ICD code to be 1000+, if we donnot limit the ICD code \
# the number will be 5000+


#test the method

# if __name__ == '__main__':
#     print is_number("20mg")


