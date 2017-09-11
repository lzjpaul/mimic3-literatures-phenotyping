import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score


def getAUC(Y, pred_res):
    fpr, tpr, thresholds = metrics.roc_curve(Y, pred_res)
    auc = metrics.auc(fpr, tpr)
    return auc


def get_macro_micro_auc(y, pred_res):
    return roc_auc_score(y, pred_res)


def get_auc_list(y, pred_res):
    macro_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res), average='macro')
    micro_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res), average='micro')
    weight_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res), average='weighted')
    average_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res))
    aucs = metrics.roc_auc_score(np.array(y), np.array(pred_res), average=None)
    return [macro_auc, micro_auc, weight_auc, average_auc, aucs], np.array(pred_res)


if __name__ == '__main__':
    print getAUC([1, 0, 1], [0, 1, 1])
