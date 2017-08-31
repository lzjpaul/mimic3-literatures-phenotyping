import numpy as np
from sklearn import linear_model
from sklearn import metrics

from baseline_method.compute_accurency import getAUC


class MultiLogistic(object):
    def __init__(self, num_logs):
        self.num_logs = num_logs
        self.models = []
        for i in range(self.num_logs):
            self.models.append(
                linear_model.LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10, 100], cv=10, tol=0.0001)
                # linear_model.LogisticRegression()
            )

    def training(self, training_x, training_y):
        assert len(training_x) == len(training_y)
        assert self.num_logs == len(training_y[0])
        for i in range(self.num_logs):
            print 'training :', i
            # print training_x.shape, training_y.shape
            self.models[i].fit(training_x, training_y[:, i])

    def testing(self, testing_x, testing_y):
        #auc_list = []
        re_list = []
        for i in range(self.num_logs):
            re_y = self.models[i].predict_proba(testing_x)[:, 1]
            # print testing_y[:, i]
            # print re_y
            re_list.append(re_y)
            #auc_list.append(getAUC(testing_y[:, i], re_y))
        print len(testing_x), self.num_logs
        print len(re_list), len(re_list[0])
        re_list = np.array(re_list).T
        macro_auc = metrics.roc_auc_score(np.array(testing_y), re_list, average='macro')
        micro_auc = metrics.roc_auc_score(np.array(testing_y), re_list, average='micro')
        weight_auc = metrics.roc_auc_score(np.array(testing_y), re_list, average='weighted')
        average_auc = metrics.roc_auc_score(np.array(testing_y), re_list)
        aucs = metrics.roc_auc_score(np.array(testing_y), re_list, average=None)
        return [macro_auc, micro_auc, weight_auc, average_auc, aucs], re_list


if __name__ == '__main__':
    test_train_x = np.random.random((30, 10))
    test_train_y = np.random.randint(low=0, high=2, size=(30, 4))
    print test_train_x
    print test_train_y
    mul = MultiLogistic(4)
    mul.training(test_train_x, test_train_y)
