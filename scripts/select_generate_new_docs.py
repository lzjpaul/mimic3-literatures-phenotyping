import pandas as pd
import numpy as np
from utility.directory_utility import Directory
from utility.csv_utility import CsvUtility
from utility.plot_utility import draw_pl
from gensim import corpora


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
    #print len(limit_new_docs)
    return limit_new_docs

def get_docs_frequence_kind_map(file_path):
    file_contend = np.array(pd.read_csv(file_path, header=None))
    doc_maps = []
    for doc in file_contend:
        doc_maps.extend([[doc[2], doc[1]]])
    # print len(doc_maps), len(doc_maps[0])
    return doc_maps



if __name__ == '__main__':
    # get_good_docs('../data-repository/result/jack_1.csv', 10, 2)
    file_list = Directory.folder_process('../data-repository/result_0.8')

    merge_dict = dict({})
    doc_map = []
    for file_path in file_list:
        dict_tmp = get_good_docs(file_path, 80, 10)
        print 'this dict len : ', len(dict_tmp)
        merge_dict.update(dict_tmp)
        print 'after the merge : ', len(merge_dict)
        doc_map.extend(get_docs_frequence_kind_map(file_path=file_path))
    draw_pl(x_y=doc_map, type='o')
    # print merge_dict
    texts = [[word for word in doc.split(' ')] for doc in merge_dict.values()]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save('../data-repository/available_word_in_literature.dict')
    print dictionary

    CsvUtility.write_dict2csv(merge_dict, '../data-repository', 'selected_docs4LDA.csv')




