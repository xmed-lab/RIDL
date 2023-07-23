import os
import numpy as np

import SimpleITK as sitk
import glob


# VERSION = 8
VERSION = 18
DCM_DIR = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/DICOMS_CRP"
SEG_DIR = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/SEG_CRP"
PYRAD_DIR = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/RADFEAT_CRP"


# feat_key_list = ['original_glcm_Idn_krn_5']
feat_key_list = ['original_glcm_Idn_krn_1', 'original_glcm_Idn_krn_2', 'original_glcm_Idn_krn_5','original_glcm_Idn_krn_10']

DCMRAD_DIR = "/home/wdaiaj/projects/cardiac_prognosis/DATA/HU_SGBX_CLEAN_DATA_STDR/DICOMSRADCAT_CRP_V{}".format(VERSION)
os.makedirs(DCMRAD_DIR, exist_ok= True)

    
dcm_dir_list = sorted(glob.glob(os.path.join(DCM_DIR, "*.nii.gz")))
print(dcm_dir_list)

for dcm_dir_idx in range(len(dcm_dir_list)):
     dcm_path_itr = dcm_dir_list[dcm_dir_idx]
     data_list = []

     dcm_dir_itr = os.path.basename(dcm_path_itr)
     ct = sitk.ReadImage(dcm_path_itr, sitk.sitkInt32)
     ct_array = sitk.GetArrayFromImage(ct)

     data_list.append(np.expand_dims(ct_array, axis = 0))

     for rad_feat_itr in feat_key_list:
          rad_path_itr = os.path.join(PYRAD_DIR, rad_feat_itr, dcm_dir_itr)
          rad_data = sitk.ReadImage(rad_path_itr, sitk.sitkFloat32)
          rad_array = sitk.GetArrayFromImage(rad_data)

          if "glcm" in rad_feat_itr:
               ### check NA
               glcm_error_dir = os.path.join(PYRAD_DIR, "nanarray_{}".format(rad_feat_itr))
               rad_array_isnan = np.load(os.path.join(glcm_error_dir, dcm_dir_itr.replace(".nii.gz", ".npy")))
               print(np.sum(rad_array_isnan.astype(int)))
               # exit()
               ###### check                
               rad_array[rad_array_isnan] = 0
          print(np.sum(rad_array), np.sum(rad_array>0))
          data_list.append(np.expand_dims(rad_array, axis = 0))
   

     seg_dir_itr = os.path.join(SEG_DIR, dcm_dir_itr)
     seg = sitk.ReadImage(seg_dir_itr, sitk.sitkInt32)
     seg_array = sitk.GetArrayFromImage(seg)
     data_list.append(np.expand_dims(seg_array, axis = 0))

     concat_array = np.concatenate(data_list, axis = 0)
     print(concat_array.shape)

     np.save(os.path.join(DCMRAD_DIR, dcm_dir_itr.replace(".nii.gz", ".npy")), concat_array)

