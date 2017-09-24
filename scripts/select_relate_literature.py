#!/usr/bin/env Python
# coding=utf-8
import os
import sys
import nltk.data
import pandas as pd
import numpy as np
import argparse
from time import clock


sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.csv_utility import CsvUtility
from utility.directory_utility import Directory
from utility.nlp_utility import NLP_Utility

class Doc2WordList(object):
    def __init__(self, doc_path, output_filter_file_path, stop_word=[]):
        self.doc_path = doc_path
        self.out_filter_file_path = output_filter_file_path
        self.stop_word = stop_word
        # don not need the word count now
        # self.word_count = {}
        self.entity_count = {}

    def get_text(self):
        ignore_line = ["==== Front", "Background", "Methods",
                       "Results", "Conclusions", "==== Body",
                       "Authors' contributions", "Click here for file",
                       "Acknowledgements"]
        stop_line = ["Figures and Tables", "==== Refs"]

        # read the text and process by lines
        rows_text = ""
        with open(self.doc_path, 'r') as f:
            rows = f.readlines()
            for row in rows:
                if len(row) == 0 or row.strip() in ignore_line: continue
                if row.strip() in stop_line: break
                rows_text += row.strip() + " "
        return rows_text

    # def doc2words(self, rows_text):
    #     words_list = CsvUtility.text2words(rows_text, stop_words=self.stop_word, stem_word=False)
    #     for word in words_list:
    #         self.word_count[word] = self.word_count[word]+1 if word in self.word_count else 1

    # def find_entity(self, entity_list):
    #     entity_set = set(entity_list)
    #     for (key, value) in self.word_count.items():
    #         if key in entity_set:
    #             self.entity_count[key] = value

    def doc2sentences(self, rows_text, token):
        sentences_list = CsvUtility.text2sentence(raw_text=rows_text, token=token, stop_words=self.stop_word, stem_word=False)
        # for sentence in sentences_list:
        #     for word in sentence:
        #         self.word_count[word] = self.word_count[word] + 1 if word in self.word_count else 1
        return sentences_list

    def find_entity_map(self, entity_map, sentences_list):
        entity_key_list = list(entity_map.keys())
        for sentence in sentences_list:
            for keywords in entity_key_list:
                if NLP_Utility.keyword_in_sentence(keyword=keywords.split(" "), sentence=sentence, percent=1):
                    entity_key = entity_map[keywords]
                    self.entity_count[entity_key] = self.entity_count[entity_key]+1 if entity_key in self.entity_count else 1

    def write_filtered_file(self):
        return CsvUtility.write_key_value_times(self.entity_count, self.out_filter_file_path, 'F_'+os.path.basename(self.doc_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the knowledge from literature dataset.")
    parser.add_argument('doc_origin_path', type=str,
                        help='Original literature directory may containing several folders.')
    parser.add_argument('output_path', type=unicode, help='Directory where output data should be written.')
    parser.add_argument('out_filter_file_path', type=str, help='if a file has entities, generate the filted file filled with the items')
    parser.add_argument('entity_file', type=unicode, help='Name of entity csv file should be read')
    args, _ = parser.parse_known_args()
    try:
        os.makedirs(args.output_path)
    except Exception:
        pass
    # use the entity vocabulary to filter the document
    # entity_list = list(pd.read_csv(os.path.join(args.output_path, args.entity_file), header=None).ix[:, 0])

    # use the maps of entity and ICD code to filter the document
    entity_df = pd.read_csv(os.path.join(args.output_path, args.entity_file), header=None)
    # entity_df = pd.read_csv(args.output_path+args.entity_file, header=None)
    print "entity map:"
    print entity_df[:5]

    entity_map = {}
    for i in range(entity_df.shape[0]):
        # assert entity_df.ix[i, 1] not in entity_map
        entity_map[entity_df.ix[i, 1]] = entity_df.ix[i, 0]

    # print 'entity size :', len(entity_list)
    print 'entity size :', len(entity_map.items())

    file_path_list = Directory.folder_process(args.doc_origin_path)
    print 'file list size : ', len(file_path_list)

    # vocabulary_count = {}
    used_entity_count = {}
    doc2entity = {}
    i_count = 0
    start_time = clock()
    # use the NLTK tokenizer to split the paragraph into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for file_path in file_path_list:
        print file_path
        doc2word_list = Doc2WordList(doc_path=file_path, output_filter_file_path=args.out_filter_file_path, stop_word=[])
        print 'step 1'
        doc_text = doc2word_list.get_text()
        print 'step 2'
        sentence_list = doc2word_list.doc2sentences(doc_text, tokenizer)
        print 'step 3'
        doc2word_list.find_entity_map(entity_map=entity_map, sentences_list=sentence_list)
        print 'step 4'
        # for (key, value) in doc2word_list.word_count.items():
        #     vocabulary_count[key] = vocabulary_count[key]+value if key in vocabulary_count else value
        for (key, value) in doc2word_list.entity_count.items():
            used_entity_count[key] = used_entity_count[key]+value if key in used_entity_count else value
        # assert file_path not in doc2entity
        if len(doc2word_list.entity_count) > 0:
            doc2entity[file_path] = str(sum(list(doc2word_list.entity_count.values()))) + "," + \
                                    str(len(doc2word_list.entity_count.items()))
            # for key_entity in doc2word_list.entity_count.keys():
            #     doc2entity[file_path] = doc2entity[file_path] + "," + key_entity
            print 'write..'
            doc2word_list.write_filtered_file()
        i_count += 1
        if i_count % 5 == 0:
            end_time = clock()
            print('\rFile Completed {0} of {1}... Spend {2} s'.format(i_count, len(file_path_list), (end_time - start_time)))
            start_time = end_time

    # print "vocabulary size : ", len(vocabulary_count)
    print "using entity size : ", len(used_entity_count)
    print "num of docs having entity : ", len(doc2entity)
    # CsvUtility.write_dict2csv(raw_dict=vocabulary_count, csv_path=args.output_path,
                              # file_name='literature_vocabulary.csv')
    CsvUtility.write_dict2csv(raw_dict=used_entity_count, csv_path=args.output_path,
                              file_name='used_entity.csv')
    CsvUtility.write_dict2csv(raw_dict=doc2entity, csv_path=args.output_path,
                              file_name='doc2entity.csv')
    print '******************************************************************************'
#test code

#python select_relate_literature.py ../data-repository/literature_doc ../data-repository ../data-repository/new_literature entity_dict.csv
