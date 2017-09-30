import pandas as pd
import numpy as np
from utility.directory_utility import Directory
from utility.csv_utility import CsvUtility


def get_good_docs(file_path, limit_count, limit_kind):
    file_contend = np.array(pd.read_csv(file_path, header=None))

    # print file_contend.shape
    # print file_contend[:5]
    # for i in range(file_contend.shape[1]):
    #     print type(file_contend[0][i])

    limit_new_docs = {}
    for doc in file_contend:
        if doc[1] >= limit_kind and doc[2] >= limit_count:
            words_list = []
            for word_dict in doc[3].strip().split(' '):
                per_word = word_dict.split(':')
                for word_count in range(int(per_word[1])):
                    words_list.append(per_word[0])
            limit_new_docs[doc[0]] = ' '.join(words_list)
    print len(limit_new_docs)
    return limit_new_docs




if __name__ == '__main__':
    get_good_docs('../data-repository/result/jack_1.csv', 80, 10)
    file_list = Directory.folder_process('../data-repository/result')

    merge_dict = dict({})
    for file_path in file_list:
        dict_tmp = get_good_docs(file_path, 100, 5)
        print 'this dict len : ', len(dict_tmp)
        merge_dict.update(dict_tmp)
        print 'after the merge : ', len(merge_dict)

    CsvUtility.write_dict2csv(merge_dict, '../data-repository', 'selected_docs4LDA.csv')




