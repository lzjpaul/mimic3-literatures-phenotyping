import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

from baseline_method.load_data import load_corpus
from baseline_method.compute_accurency import get_macro_micro_auc, get_auc_list

# Mimic Dataset
print 'loading data...'
train_x, train_y, test_x, test_y, idx = load_corpus()
print 'loading ready...'

# Hyper Parameters
input_size = len(train_x[0])
hidden_size = 500
num_classes = 2
num_epochs = 5
batch_size = 100
train_batch_num = int(np.ceil(train_x.shape[0] / (batch_size * 1.0)))
test_batch_num = int(np.ceil(test_x.shape[0] / (batch_size * 1.0)))
learning_rate = 0.001


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

all_res = []

for pred_i in range(len(train_y[0])):

    predict_index = pred_i
    trainset = np.concatenate((np.array(train_x), np.array(train_y[:, predict_index]).reshape((-1, 1))), axis=1)
    testset = np.concatenate((np.array(test_x), np.array(test_y[:, predict_index]).reshape((-1, 1))), axis=1)
    print trainset.shape, testset.shape

    # Train the Model
    for epoch in range(num_epochs):
        # for i, (images, labels) in enumerate(train_loader):
        # shuffle the train set
        idx = np.random.permutation(train_x.shape[0])
        trainset = trainset[idx]
        for i in range(train_batch_num):
            # Convert numpy array to torch Variable
            start_pos = i * batch_size
            end_pos = np.min([(i + 1) * batch_size, trainset.shape[0]])
            inputs = Variable(torch.from_numpy(trainset[start_pos:end_pos, :-1])).float()
            targets = Variable(torch.from_numpy(trainset[start_pos:end_pos, -1])).long()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1, trainset.shape[0] // batch_size, loss.data[0]))

    # Test the Model
    res = []
    for i in range(test_batch_num):
        start_pos = i * batch_size
        end_pos = np.min([(i + 1) * batch_size, testset.shape[0]])
        inputs = Variable(torch.from_numpy(testset[start_pos:end_pos, :-1])).float()
        targets = Variable(torch.from_numpy(testset[start_pos:end_pos, -1])).long()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # print outputs.data
        print predicted.numpy()
        # print targets
        res.extend(list(predicted.numpy().flatten()))

    print len(res)
    all_res.append(res)

all_res = np.array(all_res).T

print all_res
print all_res.shape

print('Accuracy of the network on the 10000 test images:', get_auc_list(test_y, all_res))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
