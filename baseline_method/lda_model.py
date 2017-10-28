from gensim import corpora, models
import pandas as pd
import numpy as np
from pprint import pprint

from utility.csv_utility import CsvUtility


def process_lda(docs_path, topic_num):
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    # print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text('../data-repository/available_word_in_literature.csv')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print corpus[:5]
    print 'number of docs:', len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num, update_every=1, chunksize=1000, passes=1)

    # lda_model.print_topics(10, 10)
    # gamma = lda_model.get_topics()
    gamma = lda_model.state.get_lambda()
    # gamma = gamma / gamma.sum(axis=0)
    print "shape of gamma :", gamma.shape
    # print gamma[:30]
    CsvUtility.write_array2csv(gamma, '../data-repository', 'gamma_from_LDA.csv')
    theta = lda_model.get_document_topics(corpus)
    print 'theta: ', theta
    print len(theta)
    print theta[0]
    return lda_model.show_topics(10, 10), theta

def convert_theta2array(topic_num, theta=[]):
    theta_array = []
    for i_doc in theta:
        per_doc_topics = [0]*topic_num
        for topics_dist in i_doc:
            per_doc_topics[topics_dist[0]] += topics_dist[1]
        theta_array.append(per_doc_topics)
    return theta_array



if __name__ == '__main__':
    # pprint(process_lda('../data-repository/selected_docs4LDA.csv', 20))
    _, theta = process_lda('../data-repository/selected_docs4LDA.csv', 20)
    theta_array = convert_theta2array(20, theta)
    # CsvUtility.write_array2csv(theta_array, '../data-repository', 'theta_array.csv')
    pd.DataFrame(theta_array).to_csv('../data-repository/theta_array.csv', sep='\t', columns=None, index=False)