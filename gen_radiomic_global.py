#https://discourse.slicer.org/t/voxel-based-radiomic-feature-extraction/18901/4
# https://github.com/AIM-Harvard/pyradiomics/issues/560


import logging
import pandas as pd
import nrrd
import os

import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

resamp = [3.0,3.0,3.0]

SOURCE_DCM_DIR = "miccai_ridl/DATA/HU_CLEAN_DATA_STDR/DICOMS_CRP128x128x64"
SOURCE_SEG_DIR = "miccai_ridl/DATA/HU_CLEAN_DATA_STDR/SEG_CRP128x128x64"

REF_DATA = "miccai_ridl/DATA/ITK/ITK_PYDCMFEAT/pydicm_feat.csv"

if resamp == [1.0,1.0,1.0]:
    OUT_DATA = "miccai_ridl/DATA/ITK/ITK_PYDCMFEAT/pydicm_feat_CRP128x128x64_HU_R1x1x1.csv"
else:
    OUT_DATA = "miccai_ridl/DATA/ITK/ITK_PYDCMFEAT/pydicm_feat_CRP128x128x64_HU_R3x3x3.csv"






if not os.path.isfile(OUT_DATA):


    ref_data_raw = pd.read_csv(REF_DATA)
    pat_list = list(ref_data_raw['pat_id'])

    feat_data_out = None

    for pat_itr_idx in range(len(pat_list)):
        print("processing {} out of {}".format(pat_itr_idx, len(pat_list)))
        pat_itr = pat_list[pat_itr_idx]

        imageName = os.path.join(SOURCE_DCM_DIR, str(pat_itr).zfill(4)+".nii.gz")
        maskName = os.path.join(SOURCE_SEG_DIR, str(pat_itr).zfill(4)+".nii.gz")

        settings = {'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': resamp, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25.0, 'symmetricalGLCM': True, 'correctMask': True}

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

        extractor.enableAllFeatures()


        print("Calculating features")
        featureVector = extractor.execute(imageName, maskName)
        
        feat_df = pd.DataFrame.from_dict(featureVector,orient='index').T
        feat_df['pat_id'] = pat_itr

        if feat_data_out is not None:
            feat_data_out = feat_data_out.append(feat_df)
        else:
            feat_data_out = feat_df

        # if pat_itr_idx > 5:
        #     break


    feat_data_out = feat_data_out.merge(ref_data_raw[['pat_id', 'CV_NUM','AF_label']], on = 'pat_id')

    feat_data_out_columns = feat_data_out.columns

    keep_columns = []
    for column_itr in feat_data_out_columns:
        if "diagnostics" not in column_itr:
            keep_columns.append(column_itr)

    feat_data_out = feat_data_out[keep_columns]


    feat_data_out.to_csv(OUT_DATA, index = False)
    # print(feat_data_out)

else:

    processed_data = pd.read_csv(OUT_DATA)
    processed_data_columns = processed_data.columns

    keep_columns = []
    for column_itr in processed_data_columns:
        if "diagnostics" not in column_itr:
            keep_columns.append(column_itr)

    processed_data = processed_data[keep_columns]
    processed_data.to_csv(OUT_DATA, index = False)