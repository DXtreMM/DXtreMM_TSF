import math
import os
import numpy as np

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import sklearn.metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from itertools import product
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from sklearn.metrics import confusion_matrix
def accuracy_per_class(predict, label):
    cm = confusion_matrix(label, predict)
    return np.diag(cm)/np.sum(cm, axis = 1), cm

# predict classification label
def get_predicted_label(recon_y):
    # get the largest probability in the two categories
    pred_label = np.argmax(recon_y,axis=1)
    # return the label only
    return np.array(pred_label)

def binary_cross_entropy(pred, target, sample_weight=None, size_average=True, pos_acc=False):
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    
    if type(sample_weight)==type(None):
        sample_weight = torch.ones_like(target)
        
    loss = -torch.sum( target * torch.log( pred + 1e-20 )*sample_weight + (1.0 - target) * torch.log( 1.0 - pred + 1e-20 )*sample_weight )
    if size_average:
        loss =  loss / pred.size()[0]
    if pos_acc:
        pos_recon = (-target * torch.log( pred + 1e-20 )).mean()
        return loss, pos_recon
    else:
        return loss

def cross_entropy(pred, target, sample_weight=None, class_acc=False):
    nc = pred.size()[1]
# #     pred = pred.view(pred.shape[0], -1)
    
    # according to nn.crossentropyloss manual, sample_weight need to have shape [c], instead of [n]
    if type(sample_weight)==type(None):
        sample_weight = torch.ones(nc).to(pred.device)
    loss = nn.CrossEntropyLoss(weight = sample_weight)
    # loss = nn.CrossEntropyLoss(reduction='none')

    # w_array = np.ones((3,3))
    # w_array[0, 1] = 0.15*sample_weight[1]
    # w_array[0, 2] = 0.5*sample_weight[2]
    # w_array[1, 0] = sample_weight[1]
    # w_array[2, 0] = sample_weight[2]
    # w_array[1, 2] = 2*sample_weight[1]
    # w_array[2, 1] = 2*sample_weight[2]

    # w_array=torch.from_numpy(w_array)
        
    # nb_cl = len(sample_weight)
    # final_mask = torch.zeros_like(pred[:, 0])
    # y_pred_max = torch.max(pred, 1)[0]
    # y_pred_max = torch.reshape(y_pred_max, (pred.size()[0], 1))
    # y_pred_max_mat = torch.eq(pred, y_pred_max)

    # one_hot_target = torch.zeros_like(pred)
    # for i  in range(len(pred)):
    #     one_hot_target[i,target[i]]=1

    # for c_p, c_t in product(range(nb_cl), range(nb_cl)):
    #     # final_mask += w_array[c_t, c_p] * (y_pred_max_mat[:, c_p]) * (torch.nn.functional.one_hot(target, num_classes=3)[:, c_t])
    #     final_mask += w_array[c_t, c_p] * (y_pred_max_mat[:, c_p]) * (one_hot_target[:, c_t])
    # loss = loss(pred, target) * final_mask

#     # pred shape = [N,C]
    output = loss(pred, target)
    # output=torch.mean(loss)

    if class_acc == True:
        pred_label =  get_predicted_label(pred.detach().cpu().numpy())
        class_acc_output, cm = accuracy_per_class(pred_label.squeeze(), target.squeeze().detach().cpu().numpy())
        return output, class_acc_output, cm
    else: 
        return output
    
    
def boostrappingCI(y_label, y_pred, model_name, N=100, nseed=123):
    n_samples = y_label.shape[0]
    np.random.seed(nseed)
    auc_list = []
    auprc_list = []
    for i in range(N):
        # random select with replacement
        select_idx = np.random.choice(n_samples, n_samples)
        if len(np.unique(y_label[select_idx])) < 2:
            # re-sample if few event cases are selected
            select_idx = np.random.choice(n_samples, n_samples)
            
        auc_ = sklearn.metrics.roc_auc_score(y_label[select_idx],y_pred[select_idx])
        
#         precision_, recall_, thresholds_ = precision_recall_curve(y_label['e'][select_idx],y_pred[select_idx])
#         f1_score_ = F1_score(precision_, recall_, beta=1.0)
        auprc_ = average_precision_score(y_label[select_idx],y_pred[select_idx])
        auc_list.append(auc_)
        auprc_list.append(auprc_)
    auc_ci = np.percentile(np.array(auc_list), [5,95])
    auprc_ci = np.percentile(np.array(auprc_list), [5,95])
    print(model_name)
    print("AUC average is:{:.3f}, 90% CI is:({:.3f}, {:.3f})".format(np.array(auc_list).mean(), auc_ci[0], auc_ci[1]))
    print("AUPRC average is:{:.3f}, 90% CI is:({:.3f}, {:.3f})".format(np.array(auprc_list).mean(), auprc_ci[0], auprc_ci[1]))
    
    return (np.array(auc_list), np.array(auprc_list))

