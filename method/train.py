import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

import numpy as np
import sys
from os import path

sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from method.define_mlp import Net

def compute_r_DKN(sita_DK, phi_KW):
    sita_phi = sita_DK.dot(phi_KW)[:, np.newaxis, :].repeat(sita_DK.shape[1], axis=1)
    eps = np.ones(sita_phi.shape) * 0.000001
    sita_phi = sita_phi + eps
    sita_DKN = sita_DK[:, :, np.newaxis].repeat(phi_KW.shape[1], axis=2)
    phi_DKW = phi_KW[np.newaxis, :, :].repeat(sita_DK.shape[0], axis=0)
    re = sita_DKN * phi_DKW / sita_phi
    return re


def compute_F_gradient(sita_DK, phi_KW, weight):
    return np.sign(weight) * np.log(sita_DK.dot(phi_KW))


def compute_F(sita_DK, phi_KW, weight, alpha):
    return -1 * np.sum(np.abs(weight) * np.log(sita_DK.dot(phi_KW)))

def update_sita(alpha, r_DKN, weight):
    r_w = np.zeros((r_DKN.shape[0], r_DKN.shape[1]))
    for k in range(r_DKN.shape[1]):
        r_w[:, k] = np.sum(r_DKN[:, k, :] * np.abs(weight), axis=1).T
    alpha_DK = alpha[np.newaxis, :].repeat(r_DKN.shape[0], axis=0)
    new_sita = (alpha_DK + r_w)
    new_sita = new_sita/(new_sita.sum(axis=1)[:, np.newaxis])
    return new_sita
    pass


def train(x_train, y_train, lda_model,
          hidden_size=128,
          num_classes=80,
          num_epochs=3,
          batchsize=10,
          learning_rate=0.001):
    input_size = len(x_train[0])
    train_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(x_train),
                                       target_tensor=torch.from_numpy(y_train))
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   num_workers=2)

    net = Net(input_size, hidden_size, num_classes)
    params = list(net.parameters())
    # Loss and Optimizer
    criterion = nn.MSELoss(size_average=False)
    # lda result
    phi_kw = lda_model.get_phi_kw()
    alpha = lda_model.get_alpha()
    print 'phi---'
    print phi_kw
    sita_dk = np.random.rand(hidden_size, lda_model.topic_num)
    sita_dk = sita_dk/(sita_dk.sum(axis=1)[:, np.newaxis])
    print '0----'
    print sita_dk
    for epoch in range(num_epochs):
        for i, data_iter in enumerate(train_loader, 0):
            # Convert numpy array to torch Variable
            data_x, data_y = data_iter
            TX = Variable(data_x).float()
            TY = Variable(data_y).float()

            # Forward + Backward + Optimize
            net.zero_grad()
            output = net(TX)
            mlp_loss = criterion(output, TY)
            print mlp_loss
            output.backward(mlp_loss)

            # ToDo:e-step
            r_DKN = compute_r_DKN(sita_DK=sita_dk, phi_KW=phi_kw)

            running_loss = 0.0
            for f_i, f in enumerate(list(net.parameters())):
                f.data.sub_(f.grad.data * learning_rate)
                if f_i == 0:
                    # ToDo:m-step
                    F_gradient = compute_F_gradient(sita_DK=sita_dk, phi_KW=phi_kw, weight=f.data.numpy())
                    print '1---'
                    print F_gradient
                    sita_dk = update_sita(alpha=alpha, r_DKN=r_DKN, weight=f.data.numpy())
                    print '2---'
                    print sita_dk
                    # ToDo: regulization
                    f.data.sub_(torch.from_numpy(F_gradient).float() * learning_rate)
                    running_loss = compute_F(sita_DK=sita_dk, phi_KW=phi_kw, weight=f.data.numpy(), alpha=alpha)
                    print '3----'
                    print running_loss
            # print loss.data
            if (i + 1) % 100 == 1:
                print 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' \
                      % (epoch + 1, num_epochs, i + 1, x_train.shape[0] / batchsize, mlp_loss.data.numpy()[0] + running_loss)
    return net, sita_dk

if __name__ == '__main__':
    # x = np.random.rand(5,10)
    # y = np.random.randint(0,1,(5,80))
    # train(x,y)
    sita = np.ones((2, 3, 4))
    phi = np.ones((2, 4))
    w = np.random.rand(3)
    re = update_sita(w, sita, phi)
    print re
    print np.random.rand(3,5)
    #print re.shape
    '''
    la_sita = sita[np.newaxis, :, :]
    print la_sita.shape

    la_sita = la_sita.repeat(3, axis=0)
    print la_sita.shape
    print la_sita
    '''


