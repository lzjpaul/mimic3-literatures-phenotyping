import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from gensim import corpora, models, utils, matutils
import numpy as np
import pandas as pd
from pprint import pprint
import sys
import os

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])



from utility.csv_utility import CsvUtility
from baseline_method.load_data import load_corpus, reload_corpus
from baseline_method.compute_accurency import get_macro_micro_auc, get_auc_list


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.relu(out)
        return out


def mlp_lda(penalty_rate=100):

    # Mimic Dataset
    print 'loading data...'
    train_x, train_y, test_x, test_y, idx = load_corpus()
    print 'loading ready...'
    print 'shape of train x:', train_x.shape
    print 'shape of train y:', train_y.shape
    print 'shape of test x:', test_x.shape
    print 'shape of test y:', test_y.shape

    # Hyper Parameters
    input_size = len(train_x[0])
    hidden_size = 128
    num_classes = 80
    num_epochs = 10
    batchsize = 10
    learning_rate = 0.01

    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print 'parameter size :'
    for para in net.parameters():
        print para.size()
        # print para


    train_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(train_x),
                                       target_tensor=torch.from_numpy(train_y))
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   num_workers=2)
    test_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(test_x),
                                      target_tensor=torch.from_numpy(test_y))
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=2)

    # Train the Model
    for epoch in range(num_epochs):
        running_loss = 0.0
        count_isntance = 0
        for i, data_iter in enumerate(train_loader, 0):
            # Convert numpy array to torch Variable
            input_train_x, input_train_y = data_iter
            inputs = Variable(input_train_x).float()
            targets = Variable(input_train_y).float()

            # get the penalty from lda model
            penalty = get_simple_inference_penalty(net)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # print 'criterion loss : ', loss
            loss = loss + penalty_rate * penalty
            # print 'penalty loss : ', (penalty_rate * penalty).data.numpy()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]/num_classes
            count_isntance += 1

            # print loss.data
            if (i + 1) % 100 == 1:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1, train_x.shape[0] / batchsize, running_loss/count_isntance))
                running_loss = 0.0
                count_isntance = 0
    print 'finish training'

    # Test the Model
    res = []
    test_loss = 0.0
    test_count = 0
    for data_iter in test_loader:
        input_test_x, input_test_y = data_iter

        outputs = net(Variable(input_test_x).float())
        targets = Variable(input_test_y).float()
        # _, predicted = torch.max(outputs.data, 1)
        predicted = outputs.data

        st_loss = criterion(outputs, targets)
        test_loss += st_loss.data[0]/num_classes
        test_count += 1
        res.extend(list(predicted.numpy()))

    # save the first parameter
    paras = net.parameters()
    for i, para4 in enumerate(paras, 0):
        if i == 0:
            para4save = para4.data.numpy()

            print 'the first parameter: ', para4save.shape
            CsvUtility.write_array2csv(para4save, '../data-repository',
                                       'temp_parameter.csv')

    # get the precision of test data
    print 'result shape:', len(res), len(res[0])
    print 'test loss:', test_loss/test_count

    auc_list, _ = get_auc_list(test_y, res)
    print'AUC List:'
    print auc_list

    # Save the Model
    # torch.save(net.state_dict(), 'model.pkl')


def get_simple_inference_penalty(net):
    # get loss from gamma with lda model
    # gamma = get_gamma_lda('../data-repository/selected_docs4LDA.csv', 20)
    gamma = CsvUtility.read_array_from_csv('../data-repository', 'gamma_result.csv')
    penalty = Variable(torch.FloatTensor([0.0]))
    gammas = Variable(torch.from_numpy(gamma)).float()
    for para_iter, para in enumerate(net.parameters()):
        if para_iter == 0:
            latent_neuron_topics = para.abs().mm(gammas)
            # print 'latent_neuron_topics : ', latent_neuron_topics
            latent_neuron_topics = latent_neuron_topics / (latent_neuron_topics.sum(dim=1).view(-1, 1))
            # print 'Norm latent_neuron_topics : ', latent_neuron_topics
            penalty = Variable(torch.FloatTensor([1.0])) / (latent_neuron_topics.max(dim=1)[0].sum())

    return penalty

def get_inference_penalty(net, hidden_size, docs_path, topic_num):
    # train the lda model
    selected_docs = pd.read_csv(docs_path, header=None, index_col=[0]).values
    print 'number of docs:', selected_docs.shape
    # print selected_docs[:5]
    texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    # pprint(texts[:5])
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text('../data-repository/available_word_in_literature.csv')
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
    dictionary.save_as_text('../data-repository/available_word_in_literature.csv')
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
    CsvUtility.write_array2csv(gamma, '../data-repository', 'gamma_from_LDA.csv')
    pprint(lda_model.show_topics(10, 10))

    # change the gamma, because the number of word is less than the number of feature
    # then insert zeros to change the size of gamma into bigger gamma with the same size
    # of (feature_size, topic_number)

    gamma_id2word = {}
    with open('../data-repository/available_word_in_literature.csv') as file:
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
    feature_index = pd.read_csv('../data-repository/feature2index.csv',
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
    CsvUtility.write_array2csv(change_index_result, '../data-repository',
                               'gamma_result.csv')
    return change_index_result


if __name__ == '__main__':

    #gamma_data = get_gamma_lda('../data-repository/selected_docs4LDA.csv', 20)
    # gamma_data = CsvUtility.read_array_from_csv('../data-repository', 'gamma_result.csv')
    mlp_lda(penalty_rate=100)
    # get_inference_penalty(0, '../data-repository/selected_docs4LDA.csv', 20)

