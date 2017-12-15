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
from os import path

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')

from utility.csv_utility import CsvUtility
from baseline_method.load_data import load_corpus, reload_corpus
from baseline_method.compute_accurency import get_macro_micro_auc, get_auc_list
from baseline_method.get_LDA_penalty import get_simple_inference_penalty



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
    num_epochs = 3
    batchsize = 10
    learning_rate = 0.001

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
    neuron_topics = np.array([])
    for epoch in range(num_epochs):
        running_loss = 0.0
        count_isntance = 0
        for i, data_iter in enumerate(train_loader, 0):
            # Convert numpy array to torch Variable
            input_train_x, input_train_y = data_iter
            inputs = Variable(input_train_x).float()
            targets = Variable(input_train_y).float()

            # get the penalty from lda model
            penalty, neuron_topics = get_simple_inference_penalty(net)
            #penalty = 0

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
                CsvUtility.write_array2csv(neuron_topics, Path + '/data-repository', 'neuron_topics_' + str(i) + '.csv')
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
            CsvUtility.write_array2csv(para4save, Path+'/data-repository',
                                       'temp_parameter.csv')

    # get the precision of test data
    print 'result shape:', len(res), len(res[0])
    print 'test loss:', test_loss/test_count

    auc_list, _ = get_auc_list(test_y, res)
    print'AUC List:'
    print auc_list

    # Save the Model
    # torch.save(net.state_dict(), 'model.pkl')




if __name__ == '__main__':

    # gamma_data = get_gamma_lda(Path+'/data-repository/selected_docs4LDA.csv', 20)
    # get_gamma_lsi(Path+'/data-repository/selected_docs4LDA.csv', 20)
    # gamma_data = CsvUtility.read_array_from_csv('../data-repository', 'gamma_result.csv')
    mlp_lda(penalty_rate=100)
    # get_inference_penalty(0, '../data-repository/selected_docs4LDA.csv', 20)

