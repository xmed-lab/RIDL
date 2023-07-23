import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import sys, os
import matplotlib.pyplot as plt

import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
import glob
import pandas as pd


DICM_ROOT_DIR = "miccai_ridl/DATA/HU_SGBX_CLEAN_DATA_STDR/DICOMS_CRP"
SEG_ROOT_DIR = "miccai_ridl/DATA/HU_SGBX_CLEAN_DATA_STDR/SEG_CRP"

OUT_ROOT_DIR = "miccai_ridl/DATA/HU_SGBX_CLEAN_DATA_STDR/RADFEAT_CRP"

REF_CV_PATH = "miccai_ridl/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv"

ref_cv_data = pd.read_csv(REF_CV_PATH)

dicm_file_list = sorted(glob.glob(os.path.join(DICM_ROOT_DIR, "*.nii.gz")))

feat_key_list = ['original_glcm_Idn']
feat_itr = feat_key_list[0]

resamp = [1.0,1.0,1.0]
KERNEL_LIST = [1,2,5,10]

SETTING_LIST = []
EXTRACTOR_LIST = []

for kern_itr_idx in range(len(KERNEL_LIST)):
     kern_itr = KERNEL_LIST[kern_itr_idx]
     print("processing kernel size {}".format(kern_itr))

     KERNEL_SIZE = kern_itr
     settings_temp = {'minimumROIDimensions': 2, 
     'minimumROISize': None, 
     'normalize': False, 
     'normalizeScale': 1, 
     'removeOutliers': None, 
     'resampledPixelSpacing': resamp, 
     'interpolator': 'sitkBSpline', 
     'preCrop': False, 
     'padDistance': 5, 
     'distances': [1], 
     'force2D': False, 
     'force2Ddimension': 0, 
     'resegmentRange': None, 
     'label': 1, 
     'additionalInfo': True, 
     'binWidth': 25.0, 
     'symmetricalGLCM': True, 
     'correctMask': True,
     'maskedKernel' : True, ### can leave this as defaults
     'kernelRadius': kern_itr} 

     extractor_0 = featureextractor.RadiomicsFeatureExtractor(**settings_temp)
     extractor_0.enableImageTypes(Original={})
     extractor_0.disableAllFeatures()



     OUT_FEAT_DIR_0 = os.path.join(OUT_ROOT_DIR, "{}_krn_{}".format(feat_itr, KERNEL_SIZE))
     os.makedirs(OUT_FEAT_DIR_0, exist_ok=True)

     if 'glcm' in feat_itr:
          OUT_FEAT_NAN_DIR_0 = os.path.join(OUT_ROOT_DIR, "nanarray_{}_krn_{}".format(feat_itr, KERNEL_SIZE))
          os.makedirs(OUT_FEAT_NAN_DIR_0, exist_ok=True)

     for file_itr_idx in range(len(dicm_file_list[:])):
          
          print("processing file {} out of {}, {}".format(file_itr_idx,len(dicm_file_list[:]), kern_itr))
          data_entry = {}

          dcm_file_itr  = dicm_file_list[file_itr_idx]
          seg_file_itr = os.path.join(SEG_ROOT_DIR, os.path.basename(dcm_file_itr))

          pat_id = int(os.path.basename(dcm_file_itr).replace(".nii.gz", ""))

          imageName = dcm_file_itr
          maskName = seg_file_itr

          msk_array = sitk.GetArrayFromImage(sitk.ReadImage(maskName))


          if feat_itr == 'original_glcm_Correlation':
               extractor_0.enableFeaturesByName(glcm=['Correlation'])
          else:
               assert False, "Check keys"
     
          out_key_name = feat_itr

          result_0 = extractor_0.execute(imageName, maskName, voxelBased=True)

          z_0 = result_0[out_key_name]

          msk_ref = sitk.ReadImage(maskName)

          rif_0 =sitk.ResampleImageFilter()
          rif_0.SetReferenceImage(msk_ref)
          rif_0.SetOutputPixelType(z_0.GetPixelID())
          rif_0.SetInterpolator(sitk.sitkNearestNeighbor)
          z_mskref_0 = rif_0.Execute(z_0)


          z_mskref_array_0 = sitk.GetArrayFromImage(z_mskref_0)

          if 'glcm' in feat_itr:
               nan_array_path_0 = os.path.join(OUT_FEAT_NAN_DIR_0, os.path.basename(dcm_file_itr)).replace(".nii.gz", ".npy")
               rad_array_isnan_0 = np.isnan(z_mskref_array_0)
               np.save(nan_array_path_0, rad_array_isnan_0)

          sitk.WriteImage(sitk.Cast(z_mskref_0, sitk.sitkFloat32), os.path.join(OUT_FEAT_DIR_0, os.path.basename(dcm_file_itr)), )


