from gensim import corpora, models, utils, matutils
import pandas as pd
import numpy as np
from pprint import pprint
from torch.autograd import Variable
import torch
import lda

from utility.csv_utility import CsvUtility
from os import path
Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')


def get_simple_inference_penalty(net):
    # get loss from gamma with lda model
    gamma = get_topicdist_lda(Path+'/data-repository/selected_docs4LDA.csv', 20)
    # gamma = CsvUtility.read_array_from_csv(Path+'/data-repository', 'gamma_result.csv')
    penalty = Variable(torch.FloatTensor([0.0]))
    gammas = Variable(torch.from_numpy(gamma)).float()
    latent_neuron_topics = np.array([])
    for para_iter, para in enumerate(net.parameters()):
        if para_iter == 0:
            latent_neuron_topics = para.abs().mm(gammas)
            # print 'latent_neuron_topics : ', latent_neuron_topics
            latent_neuron_topics = latent_neuron_topics / (latent_neuron_topics.sum(dim=1).view(-1, 1))

            # print 'Norm latent_neuron_topics : ', latent_neuron_topics
            penalty = Variable(torch.FloatTensor([1.0])) / (latent_neuron_topics.max(dim=1)[0].sum())

    return penalty, latent_neuron_topics.data.numpy()


# not finish...
def get_inference_penalty(net, hidden_size, docs_path, topic_num):
    # train the lda model
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text(Path+'/data-repository/available_word_in_literature.csv')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]
    print corpus[:5]
    print len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num, update_every=1, chunksize=1000, passes=1)

    # to inference the new doc
    # initialize the variational distribution q(theta|gamma) for the chunk
    init_gamma = utils.get_random_state(None).gamma(100., 1. / 100., (hidden_size, topic_num))
    Elogtheta = matutils.dirichlet_expectation(init_gamma)
    expElogtheta = np.exp(Elogtheta)

    converged = 0
    # Now, for each document d update that document's gamma and phi
    # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
    # Lee&Seung trick which speeds things up by an order of magnitude, compared
    # to Blei's original LDA-C code, cool!).
    for para_iter, para in enumerate(net.parameters()):
        if para_iter == 0:
            para_data = para.abs()
            for d, doc in enumerate(chunk):
                if len(doc) > 0 and not isinstance(doc[0][0], six.integer_types + (np.integer,)):
                    # make sure the term IDs are ints, otherwise np will get upset
                    ids = [int(idx) for idx, _ in doc]
                else:
                    ids = [idx for idx, _ in doc]
                cts = np.array([cnt for _, cnt in doc])
                gammad = init_gamma[d, :]
                Elogthetad = Elogtheta[d, :]
                expElogthetad = expElogtheta[d, :]
                expElogbetad = lda_model.expElogbeta[:, ids]

                # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
                # phinorm is the normalizer.
                # TODO treat zeros explicitly, instead of adding 1e-100?
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

                # Iterate between gamma and phi until convergence
                for _ in xrange(lda_model.iterations):
                    lastgamma = gammad
                    # We represent phi implicitly to save memory and time.
                    # Substituting the value of the optimal phi back into
                    # the update for gamma gives this update. Cf. Lee&Seung 2001.
                    gammad = lda_model.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                    Elogthetad = matutils.dirichlet_expectation(gammad)
                    expElogthetad = np.exp(Elogthetad)
                    phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                    # If gamma hasn't changed much, we're done.
                    meanchange = np.mean(abs(gammad - lastgamma))
                    if meanchange < lda_model.gamma_threshold:
                        converged += 1
                        break
                init_gamma[d, :] = gammad
    pass


def get_gamma_lda(docs_path, topic_num):
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text(Path+'/data-repository/available_word_in_literature.csv')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print corpus[:5]
    # print len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num,
                                update_every=1, chunksize=1000, passes=1)

    # lda_model.print_topics(10, 10)
    # gamma = lda_model.get_topics().T
    gamma = lda_model.state.get_lambda()
    gamma = (gamma / gamma.sum(axis=0)).T
    print "shape of gamma :", gamma.shape
    CsvUtility.write_array2csv(gamma, Path+'/data-repository', 'gamma_from_LDA.csv')
    pprint(lda_model.show_topics(10, 10))

    # change the gamma, because the number of word is less than the number of feature
    # then insert zeros to change the size of gamma into bigger gamma with the same size
    # of (feature_size, topic_number)

    gamma_id2word = {}
    with open(Path+'/data-repository/available_word_in_literature.csv') as file:
        line_num = file.readline()
        # print line_num
        lines_contend = file.readlines()
        for line_n in lines_contend:
            line = line_n.split("\t")
            # print line
            if len(line) > 1:
                gamma_id2word[int(line[0])] = line[1]
    print 'original gamma size: ', len(gamma_id2word)
    id_list = gamma_id2word.keys()
    # print np.array(id_list).max()

    feature_word2id = {}
    feature_index = pd.read_csv(Path+'/data-repository/feature2index.csv',
                                header=None, index_col=None)
    # print feature_index.shape
    # print feature_index[:5]
    f_i = np.array(feature_index)
    # print f_i.shape, f_i[:, 1].max()
    # np.zeros((feature_index.shape[0], gamma_data.shape[1]))
    for i in range(f_i.shape[0]):
        feature_word2id[f_i[i][0]] = int(f_i[i][1])
    print 'new feature size: ', len(feature_word2id)

    change_index_result = np.zeros((feature_index.shape[0], gamma.shape[1]))
    for i in range(gamma.shape[0]):
        new_index = feature_word2id[gamma_id2word[i]]
        for j in range(gamma.shape[1]):
            change_index_result[new_index][j] += gamma[i][j]
        if i % 1000 == 0:
            print i, 'line'
    print change_index_result[:5]
    print 'after changing the size of result: ', change_index_result.shape
    CsvUtility.write_array2csv(change_index_result, Path+'/data-repository',
                               'gamma_result.csv')
    return change_index_result


def get_topicdist_lda(docs_path, topic_num):
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text(Path+'/data-repository/available_word_in_literature.csv')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print corpus[:5]
    # print len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num,
                                update_every=1, chunksize=1000, passes=1)

    # lda_model.print_topics(10, 10)
    # gamma = lda_model.get_topics().T
    gamma = lda_model.expElogbeta
    gamma = (gamma / gamma.sum(axis=0)).T
    print "shape of gamma :", gamma.shape
    CsvUtility.write_array2csv(gamma, Path+'/data-repository', 'topics_from_LDA.csv')
    pprint(lda_model.show_topics(10, 10))

    # change the gamma, because the number of word is less than the number of feature
    # then insert zeros to change the size of gamma into bigger gamma with the same size
    # of (feature_size, topic_number)

    gamma_id2word = {}
    with open(Path+'/data-repository/available_word_in_literature.csv') as file:
        line_num = file.readline()
        # print line_num
        lines_contend = file.readlines()
        for line_n in lines_contend:
            line = line_n.split("\t")
            # print line
            if len(line) > 1:
                gamma_id2word[int(line[0])] = line[1]
    print 'original gamma size: ', len(gamma_id2word)
    id_list = gamma_id2word.keys()
    # print np.array(id_list).max()

    feature_word2id = {}
    feature_index = pd.read_csv(Path+'/data-repository/feature2index.csv',
                                header=None, index_col=None)
    # print feature_index.shape
    # print feature_index[:5]
    f_i = np.array(feature_index)
    # print f_i.shape, f_i[:, 1].max()
    # np.zeros((feature_index.shape[0], gamma_data.shape[1]))
    for i in range(f_i.shape[0]):
        feature_word2id[f_i[i][0]] = int(f_i[i][1])
    print 'new feature size: ', len(feature_word2id)

    change_index_result = np.zeros((feature_index.shape[0], gamma.shape[1]))
    for i in range(gamma.shape[0]):
        new_index = feature_word2id[gamma_id2word[i]]
        for j in range(gamma.shape[1]):
            change_index_result[new_index][j] += gamma[i][j]
        if i % 1000 == 0:
            print i, 'line'
    print change_index_result[:5]
    print 'after changing the size of result: ', change_index_result.shape
    CsvUtility.write_array2csv(change_index_result, Path+'/data-repository',
                               'topicdist_result.csv')
    return change_index_result


def get_gamma_lsi(docs_path, topic_num):
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text(Path + '/data-repository/available_word_in_literature.csv')
    print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in texts]

    corpus_model = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
    corpus_tfidf = corpus_model[corpus]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
    pprint(lsi_model.print_topics(5))


if __name__ == '__main__':
    gamma_data = get_topicdist_lda(Path + '/data-repository/selected_docs4LDA.csv', 20)

    '''
    word_topic = CsvUtility.read_array_from_csv(Path + '/data-repository', 'topics_from_LDA.csv')
    print word_topic.shape
    word_topic = (word_topic/ word_topic.sum(axis=0))
    print word_topic.sum(axis=0)
    print word_topic.sum(axis=1)
    '''
    pass