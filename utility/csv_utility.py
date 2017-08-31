# -*- coding:utf-8 -*-
import re
import nltk
import nltk.data
import pandas as pd
import os
import cPickle as pickle


class CsvUtility(object):

    @staticmethod
    def text2words(raw_text, stop_words=[], stem_word=False):

        # Remove non-letters
        text = re.sub("[^a-zA-Z]", " ", raw_text)

        # Convert words to lower case and split them
        words = text.lower().split()

        # Optionally remove stop words (false by default)
        words = [w for w in words if not w in stop_words]

        # Optionally stem words
        if stem_word:
            words = [nltk.PorterStemmer().stem(w) for w in words]
        return words

    @staticmethod
    def text2sentence(raw_text, stop_words=[], stem_word=False):

        # use the NLTK tokenizer to split the paragraph into sentences
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        raw_sentences = tokenizer.tokenize(raw_text.decode('utf8').strip())

        # loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(CsvUtility.text2words(raw_text=raw_sentence,
                                                       stop_words=stop_words,
                                                       stem_word=stem_word))
        return sentences

    @staticmethod
    def write2pickle(file_path, object, type):
        with open(file_path, type) as f:
            pickle.dump(object, f)
            f.close()

    @staticmethod
    def read_pickle(file_path, type):
        with open(file_path, type) as f:
            obj = pickle.load(f)
            f.close()
        return obj

    def write_text2csv(raw_dict, csv_path, file_name):
        pd.DataFrame.from_dict(raw_dict, orient='index').to_csv(os.path.join(csv_path, file_name), header=False)


if __name__ == '__main__':

    CsvUtility.write_text2csv(dict({'e':'r', 'w':'w'}), '../data-repository', 'test_csvutility.csv')
