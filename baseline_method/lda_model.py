from gensim import corpora, models
import pandas as pd
import numpy as np
from pprint import pprint
import lda

from utility.csv_utility import CsvUtility
from os import path
Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')

def process_lda(docs_path, topic_num):
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    # print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text(Path+'/data-repository/available_word_in_literature.csv')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print corpus[:5]
    print 'number of docs:', len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num, update_every=1, chunksize=1000, passes=1)

    # lda_model.print_topics(10, 10)
    # gamma = lda_model.get_topics()
    #gamma = lda_model.state.get_lambda()
    # gamma = gamma / gamma.sum(axis=0)
    #print "shape of gamma :", gamma.shape
    # print gamma[:30]
    #CsvUtility.write_array2csv(gamma, Path+'/data-repository', 'gamma_from_LDA.csv')
    '''
    beta = lda_model.state.sstats
    print beta.shape
    print beta[:5]
    
    return lda_model.show_topics(20, 50), beta
    '''
    for i in range(30):
        print lda_model.get_term_topics(i)

def convert_theta2array(topic_num, theta=[]):
    theta_array = []
    for i_doc in theta:
        per_doc_topics = [0]*topic_num
        for topics_dist in i_doc:
            per_doc_topics[topics_dist[0]] += topics_dist[1]
        theta_array.append(per_doc_topics)
    return theta_array

def change_topics_type(topics_from_lda):
    topics_info = {}
    for topic_line in topics_from_lda:
        words_dists = []
        for wds in topic_line[1].split(' + '):
            words_dists.append(wds.split('*'))
        topics_info[topic_line[0]] = words_dists
    return topics_info

def show_better_topics(topics_info, word_descriptions):
    with open('../data-repository/show_topics.csv', 'w') as wf:
        for items in topics_info.keys():
            wf.write(str(items) + '\n')
            word_list = topics_info[items]
            for word in word_list:
                descri = word_descriptions[word[1].replace('"', '')]
                line_text = word[0] + '\t' + word[1] + '\t' + descri + '\n'
                wf.write(line_text)




if __name__ == '__main__':
    # pprint(process_lda('../data-repository/selected_docs4LDA.csv', 20))
    process_lda(Path+'/data-repository/selected_docs4LDA.csv', 20)

    # topics_information = change_topics_type(topics)
    # entity_df = np.array(pd.read_csv('../data-repository/entity_dict.csv', header=None))
    # word_descrip_dict = {}
    # for entity_line in entity_df:
        # word_descrip_dict[entity_line[0]] = entity_line[1]
    # show_better_topics(topics_information, word_descrip_dict)
    '''
    theta_array = convert_theta2array(20, theta)
    # CsvUtility.write_array2csv(theta_array, '../data-repository', 'theta_array.csv')
    pd.DataFrame(theta_array).to_csv('../data-repository/theta_array.csv', sep='\t', columns=None, index=False)
    '''