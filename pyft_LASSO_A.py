"""

训练脚本
"""

import os
from time import time

import argparse
import pandas as pd

import numpy as np
import shutil
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

import SimpleITK as sitk


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Dataset_AF_PYDCMFEAT


# from networks.ResUNet import net
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_curve, auc



parser = argparse.ArgumentParser(description='Hyper-parameters management')


parser.add_argument('--experiment',default = 'experiment/dbg',help='Testset path')
parser.add_argument('--kd_initweight',default = None,help='Testset path')

parser.add_argument('--Ytestonly', dest='testonly', action='store_true')
parser.add_argument('--Ntestonly', dest='testonly', action='store_false')
parser.set_defaults(testonly=False)

parser.add_argument('--Ynormft', dest='normft', action='store_true')
parser.add_argument('--Nnormft', dest='normft', action='store_false')
parser.set_defaults(normft=True)



parser.add_argument('--split_seed', type=int, default=0, help='stride of sliding window')
parser.add_argument('--rand_seed', type=int, default=0, help='stride of sliding window')

parser.add_argument('--alpha_param', type=float, default=1, help='batch size') ####[1,0.1, 0.01,0.001]

parser.add_argument('--data_dir',default = 'miccai_ridl/DATA/ITK/ITK_PYDCMFEAT/pydicm_feat.csv',help='Testset path')


args = parser.parse_args()


print(args.__dict__, flush = True)


if os.path.isdir(args.experiment):
    pass
else:
    os.makedirs(args.experiment)

CV_NUM = 5

AVG_DICT = {
    'net_best.pth': {
        'AUC_AVG_LIST':[],
        'PRAVG_LIST':[],
        'F1_list':[],
        'RECALL_LIST':[],
        'PRECIS_LIST':[],
        'ACC_LIST':[]
    }
}




para_experiment_orig = args.experiment


for cv_itr in range(0, CV_NUM):
    print("Processing CV {} out of {}".format(cv_itr, CV_NUM))
    args.experiment = para_experiment_orig + "/CV{}".format(cv_itr)
    
    print("Args experiment check {}".format(args.experiment))

    
    net = Lasso(alpha=args.alpha_param, random_state=args.rand_seed, max_iter=50000)

    if os.path.isdir(args.experiment):
        pass
    else:
        os.makedirs(args.experiment)

    train_ds = Dataset_AF_PYDCMFEAT(split='train', 
                                cv_num = cv_itr, 
                                cv_num_total = CV_NUM,
                                normalize = args.normft,
                                RAW_LB_PATH = args.data_dir
                                )

    val_ds = Dataset_AF_PYDCMFEAT(split='val', 
                                cv_num = cv_itr, 
                                cv_num_total = CV_NUM,
                                normalize = args.normft,
                                RAW_LB_PATH = args.data_dir
                                )

    test_ds = Dataset_AF_PYDCMFEAT(split='test', 
                                cv_num = cv_itr, 
                                cv_num_total = CV_NUM,
                                normalize = args.normft,
                                RAW_LB_PATH = args.data_dir
                                )


    best_val_loss = 99999
    best_cls_loss = 99999
    best_avp_loss = 0
    best_auc_loss = 0
    start = time()


    if args.testonly:
        epoch_ttltrn = 0
        print("Skipping training")
    else:
        epoch_ttltrn = 1

    pred_out_file_check = os.path.join(args.experiment, "net_best_avp", "test_pred", "pred_performance_TEST.csv")

    for epoch in range(epoch_ttltrn):    
        if os.path.isfile(pred_out_file_check):
            print("file {} already exists. Skipping train".format(pred_out_file_check))
            break

        mean_loss = []
        mean_loss_val = []
        mean_acc_val = []
        mean_cls_val = []

        val_pred_score = []
        val_gt = []

        train_df_feat, train_df_label = train_ds.df_feat, train_ds.label
        train_df_feat_array = np.asarray(train_df_feat)
        train_df_label_array = np.asarray(train_df_label)


        val_df_feat, val_df_label = val_ds.df_feat, val_ds.label
        val_df_feat_array = np.asarray(val_df_feat)
        val_df_label_array = np.asarray(val_df_label)


        test_df_feat, test_df_label = test_ds.df_feat, test_ds.label
        test_df_feat_array = np.asarray(test_df_feat)
        test_df_label_array = np.asarray(test_df_label)


        net.fit(train_df_feat_array, train_df_label_array.reshape((train_df_label_array.shape[0], )))
        print(sum(net.coef_ != 0))

        coef_val = pd.Series(net.coef_, index = train_df_feat.columns)
        coef_val_abs = pd.Series(abs(net.coef_), index = train_df_feat.columns)
        coef_select = coef_val[coef_val !=0].index
        print(coef_val[coef_select])
        print(coef_val_abs[coef_select])
        print(coef_val_abs[coef_select].sort_values())

        outputs = net.predict(test_df_feat_array)
        pred_lb = (outputs.reshape(-1) > 0.5).astype(int)
    
        loss_acc = np.mean((outputs.reshape(-1) > 0.5).astype(int) == np.asarray(test_df_label_array).reshape(-1))     
            
        roc_auc_value = roc_auc_score(1 - test_df_label_array, 1 - outputs)        
        prec_avg_value = average_precision_score(1 - test_df_label_array, 1 - outputs)
        f1_avg_value = f1_score(1 - test_df_label_array, 1 - pred_lb)
        recall_value = recall_score(1 - test_df_label_array, 1 - pred_lb)
        precision_value = precision_score(1 - test_df_label_array, 1 - pred_lb)



        AVG_DICT['net_best.pth']['AUC_AVG_LIST'].append(roc_auc_value)
        AVG_DICT['net_best.pth']['PRAVG_LIST'].append(prec_avg_value)
        AVG_DICT['net_best.pth']['F1_list'].append(f1_avg_value)
        AVG_DICT['net_best.pth']['RECALL_LIST'].append(recall_value)
        AVG_DICT['net_best.pth']['PRECIS_LIST'].append(precision_value)
        AVG_DICT['net_best.pth']['ACC_LIST'].append(loss_acc)


print("AVERAGE CV AUC net_best.pth", sum(AVG_DICT['net_best.pth']['AUC_AVG_LIST'])/len(AVG_DICT['net_best.pth']['AUC_AVG_LIST']))
print("AVERAGE CV PR net_best.pth", sum(AVG_DICT['net_best.pth']['PRAVG_LIST'])/len(AVG_DICT['net_best.pth']['PRAVG_LIST']))
print("AVERAGE CV F1 net_best.pth", sum(AVG_DICT['net_best.pth']['F1_list'])/len(AVG_DICT['net_best.pth']['F1_list']))
print("AVERAGE CV Recall net_best.pth", sum(AVG_DICT['net_best.pth']['RECALL_LIST'])/len(AVG_DICT['net_best.pth']['RECALL_LIST']))
print("AVERAGE CV Recision net_best.pth", sum(AVG_DICT['net_best.pth']['PRECIS_LIST'])/len(AVG_DICT['net_best.pth']['PRECIS_LIST']))
print("AVERAGE CV ACC net_best.pth", sum(AVG_DICT['net_best.pth']['ACC_LIST'])/len(AVG_DICT['net_best.pth']['ACC_LIST']))


print("AVERAGE CV AUC net_best.pth", AVG_DICT['net_best.pth']['AUC_AVG_LIST'])
print("AVERAGE CV PR net_best.pth", AVG_DICT['net_best.pth']['PRAVG_LIST'])
print("AVERAGE CV F1 net_best.pth", AVG_DICT['net_best.pth']['F1_list'])
print("AVERAGE CV Recall net_best.pth", AVG_DICT['net_best.pth']['RECALL_LIST'])
print("AVERAGE CV Recision net_best.pth", AVG_DICT['net_best.pth']['PRECIS_LIST'])
print("AVERAGE CV ACC net_best.pth", AVG_DICT['net_best.pth']['ACC_LIST'])
print("-"*10)
