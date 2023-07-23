"""

torch中的Dataset定义脚本
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])


import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset

import pandas as pd




class Dataset_AF_BXSEG_PYRAD_SHAPE(dataset):
    def __init__(self, 
                split, 
                cv_num, 
                cv_num_total = 5,
                slice_len = 96, 
                input_size = (96,128,128),
                rndm = False, 
                flip_aug = True,
                shapeft_list = ['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice'],
                rad_version = None,
                RAW_LB_PATH = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv"):
        
        
        self.split = split
        self.cv_num = cv_num
        self.rad_version = rad_version

        cv_include_trn = []
        for cv_include_itr in range(cv_num_total - 2):
            cv_include_trn.append((cv_num + cv_include_itr + 2) % cv_num_total)

        cv_include_val = [(cv_num + 1) % cv_num_total]   
        cv_include_tst = [cv_num]


        if self.split == "train":
            cv_include = cv_include_trn
            # print(cv_include)
            # exit()
        elif self.split == "val":
            cv_include = cv_include_val
        elif self.split == "test":      
            cv_include = cv_include_tst   
        else:
            assert False, "check split value"
        
        lb_data = pd.read_csv(RAW_LB_PATH)
        # print(lb_data)

        #### get shape feats
        if RAW_LB_PATH == "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv":
            rad_feat_file_path = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/pydicm_feat_CRP128x128x64_R1x1x1.csv"
        else:
            assert False, "check RAW_LB_PATH, unsupported"

        rad_feat_data_raw = pd.read_csv(rad_feat_file_path)
        rad_feat_data_trnnorm = rad_feat_data_raw.copy()
        rad_feat_data_trnnorm = rad_feat_data_trnnorm[rad_feat_data_trnnorm['CV_NUM'].isin(cv_include_trn)]
        rad_feat_data_raw_normed = rad_feat_data_raw.copy()
        rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] = (rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] - rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].mean()) / (rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].std() + 0.001)

        print(shapeft_list)
        if len(shapeft_list) > 0:
            rad_feat_data_raw_normed_slct = rad_feat_data_raw_normed[['pat_id'] + shapeft_list]
        else:
            rad_feat_data_raw_normed_slct = rad_feat_data_raw_normed[['pat_id']]
        len_prior_merge = len(lb_data)
        lb_data = lb_data.merge(rad_feat_data_raw_normed_slct, on = 'pat_id')
        len_after_merge = len(lb_data)
        assert len_prior_merge == len_after_merge, "Something wrong with merge"        


        lb_data = lb_data[lb_data['CV_NUM'].isin(cv_include)]
        self.label = list(lb_data['AF_label'])
        self.pat_id = list(lb_data['pat_id'])

        self.shapeft_list = shapeft_list
        self.rad_glob_feat = []
        for rad_glob_feat_itr_idx in range(len(shapeft_list)):
            shapeft_itr = shapeft_list[rad_glob_feat_itr_idx]
            self.rad_glob_feat.append(list(lb_data[shapeft_itr]))

        self.ct_list = list(lb_data['orig_path'].str.replace("/home/wdaiaj/projects/cardiac_prognosis/DATA/CLEAN_DATA/DICOMS/", "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/DICOMSRADCAT_CRP_V{}/".format(2)).str.replace(".nii.gz", ".npy"))



        self.input_size = input_size
        self.rndm = rndm
        self.slice_len = slice_len
        self.flip_aug = flip_aug



    def __getitem__(self, index):

        ct_path = self.ct_list[index] + ".npy"
        ct_array_raw = np.load(ct_path.replace(".nii.gz", ".npy"))

        ct_array = ct_array_raw.astype(np.float32)
        ct_array[0] = (ct_array[0] - np.min(ct_array[0]))/(np.max(ct_array[0]) - np.min(ct_array[0])) * 2 - 1
        
        ct_array_shape = ct_array[0].shape

        if self.rndm:
            xbnd_start = np.random.randint(ct_array_shape[0] - self.input_size[0])
            ybnd_start = np.random.randint(ct_array_shape[1] - self.input_size[1])
            zbnd_start = np.random.randint(ct_array_shape[2] - self.input_size[2])

            xbnd_end = xbnd_start + self.input_size[0]
            ybnd_end = ybnd_start + self.input_size[1]
            zbnd_end = zbnd_start + self.input_size[2]


            xbnd = [xbnd_start, xbnd_end]
            ybnd = [ybnd_start, ybnd_end]
            zbnd = [zbnd_start, zbnd_end]
        else:
            xbnd_start = (ct_array_shape[0] - self.input_size[0])//2
            ybnd_start = (ct_array_shape[1] - self.input_size[1])//2
            zbnd_start = (ct_array_shape[2] - self.input_size[2])//2

            xbnd_end = xbnd_start + self.input_size[0]
            ybnd_end = ybnd_start + self.input_size[1]
            zbnd_end = zbnd_start + self.input_size[2]

            xbnd = [xbnd_start, xbnd_end]
            ybnd = [ybnd_start, ybnd_end]
            zbnd = [zbnd_start, zbnd_end]
        
        ct_array = ct_array[:, xbnd_start:xbnd_end, ybnd_start:ybnd_end, zbnd_start:zbnd_end]

        if self.flip_aug and self.rndm:
            i_flip = np.random.choice(2)
            j_flip = np.random.choice(2)
            k_flip = np.random.choice(2)

            if i_flip > 0:
                ct_array = ct_array[:, :,::-1,:].copy()
            if j_flip > 0:
                ct_array = ct_array[:, :,:,::-1].copy()
            if k_flip > 0:
                ct_array = ct_array[:, ::-1,:,:].copy()

        
        ct_array = torch.FloatTensor(ct_array)#.unsqueeze(0)

        rad_glob_feat_return = []
        for rad_glob_feat_itr_idx in range(len(self.shapeft_list)):
            rad_glob_feat_return.append(self.rad_glob_feat[rad_glob_feat_itr_idx][index])
        
        rad_glob_feat_return = torch.FloatTensor(rad_glob_feat_return)

        return ct_array, rad_glob_feat_return, self.label[index], self.pat_id[index], ct_array_raw, np.array([xbnd, ybnd, zbnd])

    def __len__(self):

        return len(self.ct_list)












class Dataset_AF_BXSEG_PYRAD_SHAPE_DUAL(dataset):
    def __init__(self, 
                split, 
                cv_num, 
                cv_num_total = 5,
                slice_len = 96, 
                input_size = (96,128,128),
                rndm = False, 
                flip_aug = True,
                shapeft_list = ['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice'],
                rad_version = None,
                RAW_LB_PATH = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv"):
        
        
        self.split = split
        self.cv_num = cv_num
        self.rad_version = rad_version

        cv_include_trn = []
        for cv_include_itr in range(cv_num_total - 2):
            cv_include_trn.append((cv_num + cv_include_itr + 2) % cv_num_total)

        cv_include_val = [(cv_num + 1) % cv_num_total]   
        cv_include_tst = [cv_num]


        if self.split == "train":
            cv_include = cv_include_trn
        elif self.split == "val":
            cv_include = cv_include_val
        elif self.split == "test":      
            cv_include = cv_include_tst   
        else:
            assert False, "check split value"
        
        lb_data = pd.read_csv(RAW_LB_PATH)

        if RAW_LB_PATH == "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv":
            rad_feat_file_path = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/pydicm_feat_CRP128x128x64_R1x1x1.csv"
        else:
            assert False, "check RAW_LB_PATH, unsupported"

        rad_feat_data_raw = pd.read_csv(rad_feat_file_path)
        rad_feat_data_trnnorm = rad_feat_data_raw.copy()
        rad_feat_data_trnnorm = rad_feat_data_trnnorm[rad_feat_data_trnnorm['CV_NUM'].isin(cv_include_trn)]
        rad_feat_data_raw_normed = rad_feat_data_raw.copy()
        rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] = (rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] - rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].mean()) / (rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].std() + 0.001)


        rad_feat_data_raw_normed_slct = rad_feat_data_raw_normed[['pat_id'] + shapeft_list]
        len_prior_merge = len(lb_data)
        lb_data = lb_data.merge(rad_feat_data_raw_normed_slct, on = 'pat_id')
        len_after_merge = len(lb_data)
        assert len_prior_merge == len_after_merge, "Something wrong with merge"  

        self.pyrad_norm_list_mean = []
        self.pyrad_norm_list_std = []

        lb_data = lb_data[lb_data['CV_NUM'].isin(cv_include)]
        self.label = list(lb_data['AF_label'])
        self.pat_id = list(lb_data['pat_id'])

        self.shapeft_list = shapeft_list
        self.rad_glob_feat = []
        for rad_glob_feat_itr_idx in range(len(shapeft_list)):
            shapeft_itr = shapeft_list[rad_glob_feat_itr_idx]
            self.rad_glob_feat.append(list(lb_data[shapeft_itr]))

        assert self.rad_version is not None, "need rad_version"

        self.ct_list = list(lb_data['orig_path'].str.replace("/home/wdaiaj/projects/cardiac_prognosis/DATA/CLEAN_DATA/DICOMS/", "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/DICOMSRADCAT_CRP_V18/").str.replace(".nii.gz", ".npy"))
        NORM_DATA_PATH = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/RADFEAT_CRP/rad_feat_stat_v{}.csv".format(18)
        norm_data = pd.read_csv(NORM_DATA_PATH)        


        self.ref_feature_list_order = ['original_glcm_Idn_krn_1', 
                                        'original_glcm_Idn_krn_2', 
                                        'original_glcm_Idn_krn_5',
                                        'original_glcm_Idn_krn_10']
        
        assert list(norm_data['feat_itr'] == self.ref_feature_list_order), "Check feature order, misaligned, will have averaging problems"
        
        for pyrad_norm_idx in range(5):
            self.pyrad_norm_list_mean.append(list(norm_data['CV{}_mean'.format(pyrad_norm_idx)]))
            self.pyrad_norm_list_std.append(list(norm_data['CV{}_std'.format(pyrad_norm_idx)]))

        self.input_size = input_size
        self.rndm = rndm
        self.slice_len = slice_len
        self.flip_aug = flip_aug



    def __getitem__(self, index):

        ct_path = self.ct_list[index] + ".npy"
        ct_array_raw_all = np.load(ct_path.replace(".nii.gz", ".npy"))

        rad_array_raw = ct_array_raw_all[[1,2,3,4]]
        ct_array_raw = ct_array_raw_all[[0,-1]]

        ct_array = ct_array_raw.astype(np.float32)
        rad_array = rad_array_raw.astype(np.float32)
        ct_array[0] = (ct_array[0] - np.min(ct_array[0]))/(np.max(ct_array[0]) - np.min(ct_array[0])) * 2 - 1

        ct_array_shape = ct_array[0].shape

        if self.rndm:
            xbnd_start = np.random.randint(ct_array_shape[0] - self.input_size[0])
            ybnd_start = np.random.randint(ct_array_shape[1] - self.input_size[1])
            zbnd_start = np.random.randint(ct_array_shape[2] - self.input_size[2])

            xbnd_end = xbnd_start + self.input_size[0]
            ybnd_end = ybnd_start + self.input_size[1]
            zbnd_end = zbnd_start + self.input_size[2]


            xbnd = [xbnd_start, xbnd_end]
            ybnd = [ybnd_start, ybnd_end]
            zbnd = [zbnd_start, zbnd_end]
        else:
            xbnd_start = (ct_array_shape[0] - self.input_size[0])//2
            ybnd_start = (ct_array_shape[1] - self.input_size[1])//2
            zbnd_start = (ct_array_shape[2] - self.input_size[2])//2

            xbnd_end = xbnd_start + self.input_size[0]
            ybnd_end = ybnd_start + self.input_size[1]
            zbnd_end = zbnd_start + self.input_size[2]

            xbnd = [xbnd_start, xbnd_end]
            ybnd = [ybnd_start, ybnd_end]
            zbnd = [zbnd_start, zbnd_end]
        
        ct_array = ct_array[:, xbnd_start:xbnd_end, ybnd_start:ybnd_end, zbnd_start:zbnd_end]
        rad_array = rad_array[:, xbnd_start:xbnd_end, ybnd_start:ybnd_end, zbnd_start:zbnd_end]

        if self.flip_aug and self.rndm:
            i_flip = np.random.choice(2)
            j_flip = np.random.choice(2)
            k_flip = np.random.choice(2)

            if i_flip > 0:
                ct_array = ct_array[:, :,::-1,:].copy()
                rad_array = rad_array[:, :,::-1,:].copy()
            if j_flip > 0:
                ct_array = ct_array[:, :,:,::-1].copy()
                rad_array = rad_array[:, :,:,::-1].copy()
            if k_flip > 0:
                ct_array = ct_array[:, ::-1,:,:].copy()
                rad_array = rad_array[:, ::-1,:,:].copy()

        
        ct_array = torch.FloatTensor(ct_array)#.unsqueeze(0)
        rad_array = torch.FloatTensor(rad_array)#.unsqueeze(0)

        rad_glob_feat_return = []
        for rad_glob_feat_itr_idx in range(len(self.shapeft_list)):
            rad_glob_feat_return.append(self.rad_glob_feat[rad_glob_feat_itr_idx][index])
        
        rad_glob_feat_return = torch.FloatTensor(rad_glob_feat_return)

        return ct_array, rad_array, rad_glob_feat_return, self.label[index], self.pat_id[index], ct_array_raw, np.array([xbnd, ybnd, zbnd])

    def __len__(self):

        return len(self.ct_list)







class Dataset_AF_PYDCMFEAT(dataset):
    def __init__(self, 
                split, 
                cv_num, 
                cv_num_total = 5,
                normalize = False,
                use_valset = False,
                RAW_LB_PATH = "/home/wdaiaj/projects/cardiac_prognosis/DATA/ITK/ITK_PYDCMFEAT/pydicm_feat.csv"):
        
        
        self.split = split


        cv_include_train = []
        if use_valset:
            for cv_include_itr in range(cv_num_total - 2):
                cv_include_train.append((cv_num + cv_include_itr + 2) % cv_num_total)
        else:
            for cv_include_itr in range(cv_num_total - 1):
                cv_include_train.append((cv_num + cv_include_itr + 1) % cv_num_total)

        cv_include_val = [(cv_num + 1) % cv_num_total]
        cv_include_test = [cv_num]


        if self.split == "train":
            cv_include = cv_include_train
        elif self.split == "val":
            cv_include = cv_include_val
        elif self.split == "test":            
            cv_include = cv_include_test
        else:
            assert False, "check split value"

        lb_data_all = pd.read_csv(RAW_LB_PATH)
        
        lb_data_train = lb_data_all[lb_data_all['CV_NUM'].isin(cv_include_train)]
        self.df_feat_train = lb_data_train.loc[:, ~lb_data_train.columns.isin(['pat_id','CV_NUM','AF_label'])]
        self.df_feat_train_mean = self.df_feat_train.mean()
        self.df_feat_train_std = self.df_feat_train.std() + 0.0001

        lb_data = lb_data_all[lb_data_all['CV_NUM'].isin(cv_include)]
        self.label = lb_data[['AF_label']]

        self.df_feat = lb_data.loc[:, ~lb_data.columns.isin(['pat_id','CV_NUM','AF_label'])]
        if normalize:
            self.df_feat=(self.df_feat - self.df_feat_train_mean)/self.df_feat_train_std

    def __getitem__(self, index):
        index = index % len(self.df)

        label = self.df_label.iloc[index]
        features = self.df_feat.iloc[index]

        
        return np.asarray(features).astype("float32"), np.asarray(label).astype("long")

    def __len__(self):

        return len(self.label)

















class Dataset_AF_PYRAD_SHAPE(dataset):
    def __init__(self, 
                split, 
                cv_num, 
                cv_num_total = 5,
                rndm = False, 
                flip_aug = True,
                shapeft_list = ['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice'],
                RAW_LB_PATH = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv"):
        
        
        self.split = split
        self.cv_num = cv_num

        cv_include_trn = []
        for cv_include_itr in range(cv_num_total - 2):
            cv_include_trn.append((cv_num + cv_include_itr + 2) % cv_num_total)

        cv_include_val = [(cv_num + 1) % cv_num_total]   
        cv_include_tst = [cv_num]


        if self.split == "train":
            cv_include = cv_include_trn
            # print(cv_include)
            # exit()
        elif self.split == "val":
            cv_include = cv_include_val
        elif self.split == "test":      
            cv_include = cv_include_tst   
        else:
            assert False, "check split value"
        
        lb_data = pd.read_csv(RAW_LB_PATH)
        # print(lb_data)

        #### get shape feats
        if RAW_LB_PATH == "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv":
            rad_feat_file_path = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/pydicm_feat_CRP128x128x64_R1x1x1.csv"
        else:
            assert False, "check RAW_LB_PATH, unsupported"

        rad_feat_data_raw = pd.read_csv(rad_feat_file_path)
        rad_feat_data_trnnorm = rad_feat_data_raw.copy()
        rad_feat_data_trnnorm = rad_feat_data_trnnorm[rad_feat_data_trnnorm['CV_NUM'].isin(cv_include_trn)]
        rad_feat_data_raw_normed = rad_feat_data_raw.copy()
        rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] = (rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] - rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].mean()) / (rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].std() + 0.001)
        # print(rad_feat_data_trnnorm.mean()[['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice']])
        # print(rad_feat_data_trnnorm.std()[['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice']])
        # print(rad_feat_data_raw)
        # print(rad_feat_data_raw_normed)
        # exit()

        rad_feat_data_raw_normed_slct = rad_feat_data_raw_normed[['pat_id'] + shapeft_list]
        len_prior_merge = len(lb_data)
        lb_data = lb_data.merge(rad_feat_data_raw_normed_slct, on = 'pat_id')
        len_after_merge = len(lb_data)
        assert len_prior_merge == len_after_merge, "Something wrong with merge"        
        # print(lb_data)
        # exit()

        lb_data = lb_data[lb_data['CV_NUM'].isin(cv_include)]
        self.label = list(lb_data['AF_label'])
        self.pat_id = list(lb_data['pat_id'])

        self.shapeft_list = shapeft_list
        self.rad_glob_feat = []
        for rad_glob_feat_itr_idx in range(len(shapeft_list)):
            shapeft_itr = shapeft_list[rad_glob_feat_itr_idx]
            self.rad_glob_feat.append(list(lb_data[shapeft_itr]))

        self.rndm = rndm
        self.flip_aug = flip_aug



    def __getitem__(self, index):

        rad_glob_feat_return = []
        for rad_glob_feat_itr_idx in range(len(self.shapeft_list)):
            rad_glob_feat_return.append(self.rad_glob_feat[rad_glob_feat_itr_idx][index])
        
        rad_glob_feat_return = torch.FloatTensor(rad_glob_feat_return)

        return  rad_glob_feat_return, self.label[index], self.pat_id[index]

    def __len__(self):

        return len(self.label)















class Dataset_AF_PYRAD_SHAPE_DEEP(dataset):
    def __init__(self, 
                split, 
                cv_num, 
                cv_num_total = 5,
                rndm = False, 
                flip_aug = True,
                shapeft_list = ['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice'],
                RAW_LB_PATH = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv"):
        
        
        self.split = split
        self.cv_num = cv_num

        cv_include_trn = []
        for cv_include_itr in range(cv_num_total - 2):
            cv_include_trn.append((cv_num + cv_include_itr + 2) % cv_num_total)

        cv_include_val = [(cv_num + 1) % cv_num_total]   
        cv_include_tst = [cv_num]


        if self.split == "train":
            cv_include = cv_include_trn
            # print(cv_include)
            # exit()
        elif self.split == "val":
            cv_include = cv_include_val
        elif self.split == "test":      
            cv_include = cv_include_tst   
        else:
            assert False, "check split value"
        
        lb_data = pd.read_csv(RAW_LB_PATH)
        # print(lb_data)

        #### get shape feats
        if RAW_LB_PATH == "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv":
            rad_feat_file_path = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/pydicm_feat_CRP128x128x64_R1x1x1.csv"
        else:
            assert False, "check RAW_LB_PATH, unsupported"

        rad_feat_data_raw = pd.read_csv(rad_feat_file_path)
        rad_feat_data_trnnorm = rad_feat_data_raw.copy()
        rad_feat_data_trnnorm = rad_feat_data_trnnorm[rad_feat_data_trnnorm['CV_NUM'].isin(cv_include_trn)]
        rad_feat_data_raw_normed = rad_feat_data_raw.copy()
        rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] = (rad_feat_data_raw_normed.loc[:, ~rad_feat_data_raw_normed.columns.isin(['pat_id'])] - rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].mean()) / (rad_feat_data_trnnorm.loc[:, ~rad_feat_data_trnnorm.columns.isin(['pat_id'])].std() + 0.001)
        # print(rad_feat_data_trnnorm.mean()[['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice']])
        # print(rad_feat_data_trnnorm.std()[['original_shape_Maximum3DDiameter', 'original_shape_Maximum2DDiameterSlice']])
        # print(rad_feat_data_raw)
        # print(rad_feat_data_raw_normed)
        # exit()

        rad_feat_data_raw_normed_slct = rad_feat_data_raw_normed[['pat_id'] + shapeft_list]
        len_prior_merge = len(lb_data)
        lb_data = lb_data.merge(rad_feat_data_raw_normed_slct, on = 'pat_id')
        len_after_merge = len(lb_data)
        assert len_prior_merge == len_after_merge, "Something wrong with merge"        
        # print(lb_data)
        # exit()

        lb_data = lb_data[lb_data['CV_NUM'].isin(cv_include)]
        self.label = list(lb_data['AF_label'])
        self.pat_id = list(lb_data['pat_id'])

        self.shapeft_list = shapeft_list
        self.rad_glob_feat = []
        for rad_glob_feat_itr_idx in range(len(shapeft_list)):
            shapeft_itr = shapeft_list[rad_glob_feat_itr_idx]
            self.rad_glob_feat.append(list(lb_data[shapeft_itr]))

        self.rndm = rndm
        self.flip_aug = flip_aug



    def __getitem__(self, index):

        rad_glob_feat_return = []
        for rad_glob_feat_itr_idx in range(len(self.shapeft_list)):
            rad_glob_feat_return.append(self.rad_glob_feat[rad_glob_feat_itr_idx][index])
        

        pat_id_index = self.pat_id[index]
        # pat_deep_feat_path = os.path.join("/home/wdaiaj/projects/cardiac_prognosis/code/experiment/230125_HUSGBXSegCBAMfc1mtwcl1wrec1_in2ou2_btch1_epc100_s0/net_best_cls/test_feat", "{}_feat.npy".format(str(pat_id_index).zfill(4)))
        pat_deep_feat_path = os.path.join("/home/wdaiaj/projects/cardiac_prognosis/code/experiment/230220_RADV19_FTFUSCAM_GLBFTMLT0_HUSGBXSegCBAMfc2mtwcl1wrec1wradec0_in2ou4_btch1_epc100_s5/net_best_cls/test_feat", "{}_feat.npy".format(str(pat_id_index).zfill(4)))

        pat_deep_data = np.load(pat_deep_feat_path)
        # print(pat_deep_data.shape)
        # exit()

        pat_deep_data_tensor = torch.FloatTensor(pat_deep_data)
        rad_glob_feat_tensor = torch.FloatTensor(rad_glob_feat_return)

        rad_glob_feat_return = torch.cat((rad_glob_feat_tensor, pat_deep_data_tensor), dim = -1)


        return  rad_glob_feat_return, self.label[index], self.pat_id[index]

    def __len__(self):

        return len(self.label)



