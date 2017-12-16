
import numpy as np
import pandas as pd
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer

from utility.csv_utility import CsvUtility
from os import path

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

Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')



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


original_path = 'E:\medical data\MIMICIII_data'

def get_revert_prescription():
    prescription_df = pd.read_csv(os.path.join(Path, 'MIMICIII_data/PRESCRIPTIONS.csv'), dtype=str)
    drug_df = CsvUtility.read_pickle(Path+'/data-repository/prescription_drug_over.pkl', 'r')
    # print type(list(drug_df.index)[0])
    # print np.array(list(drug_df.index), dtype=str)
    print prescription_df.shape
    print prescription_df[:5]
    print prescription_df.dtypes
    print prescription_df.describe()
    prescription_dict = prescription_df[['FORMULARY_DRUG_CD', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC']]
    print prescription_dict.shape
    prescription_dict = prescription_dict.dropna()
    print prescription_dict.shape
    prescription_dict = prescription_dict.drop_duplicates()
    print prescription_dict.shape

    # print prescription_dict[:5]
    # prescription_dict.to_csv("../data-repository/prescription_dict.csv", index=None)

    stop_char = ['(', ')', '/', '/"', '-']
    stop_str = {"*nf*", "a", "b","of", "and", "by", "to", "or", "the", "in", "with"
                     , "not", "classified", "for", "on", "from", "without"
                     , "as", "other", "than", "more", "at", "one", "all"
                     , "its", "may", "after", "any", "d", "be", "into"
                     , "their", "which", "an", "ec", "c", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"
                     , "q", "i", "s", "t", "u", "v", "w", "x", "y", "z", "vs.", "mg", "extended-release", ""}
    revert_prescrip_dict = {}
    prescrip_list = prescription_dict.values
    print prescrip_list[:5]
    for i in range(len(prescrip_list)):
        if prescrip_list[i][0] in list(drug_df.index):
            word_list_tmp = []
            prescrip_str = remove_bracket_from_str(prescrip_list[i][1])
            for stop_c in stop_char:
                prescrip_str = prescrip_str.replace(stop_c, ' ').strip()
            for word_tmp in prescrip_str.split(" "):
                tmp = word_tmp.lower()
                if len(tmp) > 0 and any(char.isalpha() for char in tmp):
                    if tmp.endswith("mg") and len(tmp) > 2 and is_number(tmp[:-2]):
                        pass
                    elif tmp not in stop_str:
                        word_list_tmp.append(tmp.strip())
            words = " ".join(word_list_tmp).strip()
            if len(words) > 0 and words not in revert_prescrip_dict:
                revert_prescrip_dict[words] = prescrip_list[i][0]

            word_list_tmp = []
            prescrip_str = remove_bracket_from_str(prescrip_list[i][2])
            for stop_c in stop_char:
                prescrip_str = prescrip_str.replace(stop_c, ' ').strip()
            for word_tmp in prescrip_str.split(" "):
                tmp = word_tmp.lower()
                if len(tmp) > 0 and any(char.isalpha() for char in tmp):
                    if tmp.endswith("mg") and len(tmp) > 2 and is_number(tmp[:-2]):
                        pass
                    elif tmp not in stop_str:
                        word_list_tmp.append(tmp.strip())
            words = " ".join(word_list_tmp).strip()
            if len(words) > 0 and words not in revert_prescrip_dict:
                revert_prescrip_dict[words] = prescrip_list[i][0]
    print revert_prescrip_dict
    print len(revert_prescrip_dict)

    CsvUtility.write_dict2csv(dict(revert_prescrip_dict), Path+"/data-repository", 'revert_prescription_dict.csv')

# labtest revert dict, same to prescription
def get_revert_labtest():
    labtest_df = pd.read_csv(os.path.join(Path, 'MIMICIII_data/D_LABITEMS.csv'), dtype=str)
    item_df = CsvUtility.read_pickle(Path+'/data-repository/lab_item_over.pkl', 'r')
    print item_df[:5]
    print type(list(item_df.index)[0])
    print labtest_df.shape
    print labtest_df[:5]
    print labtest_df.dtypes
    print labtest_df.describe()
    labtest_dict = labtest_df[['ITEMID', 'LABEL']]
    print labtest_dict.shape
    labtest_dict = labtest_dict.dropna()
    print labtest_dict.shape
    labtest_dict = labtest_dict.drop_duplicates()
    print labtest_dict.shape
    print labtest_dict[:5]
    # labtest_dict.to_csv("../data-repository/labtest_dict.csv", index=None)

    labtest_list = labtest_dict.values
    print labtest_list[:5]
    # print np.array(list(item_df.index), dtype=str)
    revert_labtest_dict = {}
    for i in range(len(labtest_list)):
        if labtest_list[i][0] in np.array(list(item_df.index), dtype=str):
            temp_str = remove_bracket_from_str(labtest_list[i][1])
            temp_str = remove_quotation_from_str(temp_str)
            temp_str = temp_str.replace(",", " ").strip().lower()
            revert_labtest_dict[temp_str] = labtest_list[i][0]

    print revert_labtest_dict
    print len(revert_labtest_dict)
    CsvUtility.write_dict2csv(dict(revert_labtest_dict), Path+"/data-repository", "revert_labtest_dict.csv")

# why limit the ICD code to 3, and that will make the number of ICD code to be 1000+, if we donnot limit the ICD code \
# the number will be 5000+
# this process is to merge the selected ICD code and the long title description

def get_revert_diagnoses_procedures():
    word_count = {}
    stop_list = {"of", "and", "by", "to", "or", "the", "in", "with"
        , "not", "classified", "for", "on", "from", "without"
        , "as", "other", "than", "more", "at", "one", "all"
        , "a", "its", "may", "after", "any", "d", "be", "into"
        , "their", "which", "an", "*nf", "nf*", "but", "but"
        , "", "-", "c", "c-c", "w", "e", "o", "b", "m", "g"
        , "s", "h", "t-t", "un", "ve", "k", "u", "j", "t"
        , "n"}
    diagnoses_df = CsvUtility.read_pickle(Path+'/data-repository/icd_diagnoses_over.pkl', 'r')
    procedures_df = CsvUtility.read_pickle(Path+'/data-repository/icd_procedures_over.pkl', 'r')
    data_diagnoses = pd.read_csv(os.path.join(Path, 'MIMICIII_data/D_ICD_DIAGNOSES.csv'), dtype=str)[["ICD9_CODE", "LONG_TITLE"]]
    data_procedures = pd.read_csv(os.path.join(Path, 'MIMICIII_data/D_ICD_PROCEDURES.csv'), dtype=str)[["ICD9_CODE", "LONG_TITLE"]]
    data_diagnoses.set_index(["ICD9_CODE"], inplace=True)
    data_procedures.set_index(["ICD9_CODE"], inplace=True)
    print diagnoses_df[:5]
    print diagnoses_df.shape
    print procedures_df[:5]
    print procedures_df.shape
    print data_diagnoses[:5]
    print data_diagnoses.shape
    print data_procedures[:5]
    print data_procedures.shape

    merge_diagnoses = pd.merge(diagnoses_df, data_diagnoses, how='inner', left_index=True, right_index=True)
    print merge_diagnoses[:10]
    print merge_diagnoses.shape

    merge_procedures = pd.merge(procedures_df, data_procedures, how='inner', left_index=True, right_index=True)
    print merge_procedures[:10]
    print merge_procedures.shape

    #combine the dianoses and procedures dataframe
    ICD_merge = pd.concat([merge_diagnoses,merge_procedures], axis=0)
    print ICD_merge[:5]

    icd_merge_list = np.array(ICD_merge.reset_index(), dtype=str)
    print icd_merge_list[:5]
    revert_diagnoses_procedures = {}
    for i in range(len(icd_merge_list)):
        wordlist = [re.sub("[^a-zA-Z-]", "", x.lower()) for x in icd_merge_list[i][2].split(' ')
                                if re.sub("[^a-zA-Z-]", "", x.lower()) not in stop_list]
        revert_diagnoses_procedures[" ".join(wordlist)] = icd_merge_list[i][0]
        for word in wordlist:
            word_count[word] = word_count[word] + 1 if word in word_count else 1
    CsvUtility.write_dict2csv(revert_diagnoses_procedures, Path+'/data-repository/', 'revert_diagnoses_procedures.csv')
    # CsvUtility.write_text2csv(word_count, '../data-repository/', 'revert_ICD_word_dict.csv')
    with open(Path+"/data-repository/revert_ICD_word_dict.csv", 'w') as w:
            for (key, value) in sorted(word_count.items(), key=lambda s: s[1], reverse=True):
                w.write(key + "," + str(value) + "\n")

def show_df(df, num=5):
    print df[:num]
    print 'shpe: ', df.shape

#test the method
def get_final_word_dict():
    MIMIC_word_dict = list(CsvUtility.read_pickle(Path+'/data-repository/event_instance_dict.pkl', 'r'))
    print MIMIC_word_dict[:10]
    print len(MIMIC_word_dict)
    diag_num = 0
    lab_num = 0
    drug_num = 0
    other_num = 0
    new_MIMIC_dict = {}

    for item in MIMIC_word_dict:
        if item.startswith("d_"):
            diag_num += 1
        elif item.startswith("l_"):
            lab_num += 1
        elif item.startswith("p_"):
            drug_num += 1
        else:
            other_num += 1
            print item
        new_MIMIC_dict[item[2:]] = item
    new_MIMIC_dict_df = pd.DataFrame.from_dict(dict(new_MIMIC_dict), orient='index')
    show_df(new_MIMIC_dict_df, 10)

    print 'diagnoses number :', diag_num, 'labtest number:', lab_num,'drug number:', drug_num,'other number:', other_num

    revert_diag_proce_df = pd.read_csv(Path+'/data-repository/revert_diagnoses_procedures.csv', header=None, dtype=str)
    revert_labtest_df = pd.read_csv(Path+'/data-repository/revert_labtest_dict.csv', header=None, dtype=str)
    revert_prescrip_df = pd.read_csv(Path+'/data-repository/revert_prescription_dict.csv', header=None, dtype=str)

    show_df(revert_diag_proce_df, 10)
    show_df(revert_labtest_df, 10)
    show_df(revert_prescrip_df, 10)

    concat_dict = pd.concat([revert_diag_proce_df, revert_labtest_df, revert_prescrip_df], axis=0, ignore_index=True)
    show_df(concat_dict, 20)
    concat_dict.set_index(keys=[1], inplace=True)
    show_df(concat_dict, 10)
    print len(set(list(concat_dict.index)))

    merge_df = pd.merge(new_MIMIC_dict_df, concat_dict, how='left', left_index=True, right_index=True)
    show_df(merge_df, 10)

    print len(set(list(merge_df.index)))
    print len(merge_df['0_x'].unique())
    print len(merge_df['0_y'].unique())

    merge_df.drop_duplicates()
    show_df(merge_df)
    merge_df.to_csv(Path+'/data-repository/entity_dict.csv', header=None, index=None)
    # here we get :
    # from mimic : diagnoses number : 3208 labtest number: 174 drug number: 749 other number: 220
    # from revert descriptions we get diagnoses number : 4356 labtest number: 174 drug number: 1434
    # Problem 1: here exist the maps from one CODE to multiple DESCRIPTIONS, that is allowed
    # After merge here also exist multiple CODE map to ONE CODE
    # In the end, we get 4350 words, and 5163 descriptions

if __name__ == '__main__':
    # second step:
    get_revert_prescription()
    get_revert_labtest()
    get_revert_diagnoses_procedures()
    get_final_word_dict()



