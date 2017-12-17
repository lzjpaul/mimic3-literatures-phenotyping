from os import path
import pandas as pd
import numpy as np
import pprint
from gensim import corpora, models, utils, matutils

Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')


class LdaTools(object):
    def __init__(self, doc_path=Path+'/data-repository/selected_docs4LDA.csv', topic_num=20):
        self.doc_path = doc_path
        self.topic_num = topic_num

    def train_lda(self, passes=1):
        selected_docs = pd.read_csv(self.doc_path, header=None, index_col=[0]).values
        texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.save_as_text(Path+'/data-repository/available_word_in_literature.csv')
        print self.dictionary
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        print 'number of docs:', len(corpus)
        self.lda_model = models.LdaModel(corpus, id2word=self.dictionary, num_topics=self.topic_num, passes=passes)
        print 'lda training end.'

    def get_alpha(self):
        return self.lda_model.alpha

    def get_phi_kw(self):
        lda_phi = self.lda_model.get_topics()
        return self.__change_word_index__(lda_phi, 1,
                                          Path+'/data-repository/feature2index.csv')

    def get_wk(self):
        gamma = self.lda_model.expElogbeta
        gamma_norm = (gamma / gamma.sum(axis=0)).T
        return self.__change_word_index__(gamma_norm, 0,
                                          Path + '/data-repository/feature2index.csv')

    def show_topics(self, topic_n, word_n):
        print(self.lda_model.show_topics(topic_n, word_n))

    def __change_word_index__(self, gamma, idx, feature2index_path):
        print 'dictionary size: ', self.dictionary.__len__()
        feature_word2id = {}
        feature_index = pd.read_csv(feature2index_path,header=None, index_col=None)
        f_i = np.array(feature_index)
        for i in range(f_i.shape[0]):
            feature_word2id[f_i[i][0]] = int(f_i[i][1])
        print 'feature size: ', len(feature_word2id)

        if idx == 0:
            change_index_result = np.zeros((feature_index.shape[0], gamma.shape[1]))
            for i in range(gamma.shape[0]):
                new_index = feature_word2id[self.dictionary.__getitem__(i)]
                for j in range(gamma.shape[1]):
                    change_index_result[new_index][j] += gamma[i][j]
                if i % 1000 == 0:
                    print i, 'line'
            print 'after changing the size of result: ', change_index_result.shape
        else:
            change_index_result = np.zeros((gamma.shape[0], feature_index.shape[0]))
            for j in range(gamma.shape[1]):
                new_index = feature_word2id[self.dictionary.__getitem__(j)]
                for i in range(gamma.shape[0]):
                    change_index_result[i][new_index] += gamma[i][j]
                if j % 1000 == 0:
                    print j, 'line'
            print 'after changing the size of result: ', change_index_result.shape
        return change_index_result


if __name__ == '__main__':

    lt = LdaTools(Path+'/data-repository/selected_docs4LDA.csv', 20)
    lt.train_lda(1)
    print lt.get_alpha()
    print lt.get_phi_kw()
    print lt.get_wk()

