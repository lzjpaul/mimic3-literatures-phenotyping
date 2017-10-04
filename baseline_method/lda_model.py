from gensim import corpora, models
import pandas as pd
import numpy as np
from pprint import pprint


def process_lda(docs_path, topic_num):
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    print selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save('../data-repository/available_word_in_literature.dict')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print corpus[:5]
    print len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num, update_every=1, chunksize=1000, passes=1)

    lda_model.print_topics(10, 10)
    return lda_model.show_topics(10, 10)


if __name__ == '__main__':
    pprint(process_lda('../data-repository/selected_docs4LDA.csv', 20))
