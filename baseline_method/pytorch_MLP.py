import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from gensim import corpora, models
import numpy as np
import pandas as pd

from utility.csv_utility import CsvUtility
from baseline_method.load_data import load_corpus, reload_corpus
from baseline_method.compute_accurency import get_macro_micro_auc, get_auc_list

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True,)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.relu(out)
        return out

def mlp_lda():

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
    hidden_size = 100
    num_classes = 80
    num_epochs = 1
    batchsize = 10
    learning_rate = 0.001

    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print 'parameter size :'
    for para in net.parameters():
        print para.size()


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

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(inputs)
            loss = criterion(outputs, targets)
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
        # total += labels.size(0)
        # print outputs.data
        # print predicted.numpy().shape
        # print targets
        # minus = predicted.numpy()-input_test_y
        # t_loss = minus.T.dot(minus).sum()
        # t_loss = t_loss / (minus.shape[0]*minus.shape[1])
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
    print len(corpus)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num,
                                update_every=1, chunksize=1000, passes=1)

    # lda_model.print_topics(10, 10)
    gamma = lda_model.get_topics()
    print "shape of gamma :", gamma.shape
    CsvUtility.write_array2csv(gamma, '../data-repository', 'gamma_from_LDA.csv')
    return lda_model.show_topics(10, 10)

if __name__ == '__main__':
    mlp_lda()
