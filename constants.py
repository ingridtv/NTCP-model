"""
Created on 30/09/2020
@author: ingridtveten
"""

from os.path import join
import numpy as np

import SETUP_LOCAL as setup

# ===========================
#   CONSTANTS
# ===========================
NUM_DECIMALS_SLICE_LOCATION = 3

ALL_PATIENTS = np.arange(1, 260, 1, dtype=int)
NO_PATIENT_DATA = (2, 16, 26, 55, 65, 77, 78, 91, 162, 165, 168, 172, 174, 183,
                   200, 213, 220, 222, 226, 245, 246, 248, 250, 255, 260)
AVAILABLE_PATIENTS = np.asarray([id for id in ALL_PATIENTS
                                 if (id not in NO_PATIENT_DATA)])

VERIFIED_FOLLOW_UP = (5, 12, 18, 24, 36)
UNVERIFIED_FOLLOW_UP = (30, 42, 48, 54, 60)


# ===========================
#   FILE & PATH CONSTANTS
# ===========================

PROJECT_FILE_PATH = setup.LOCAL_PROJECT_FILE_PATH
PROJECT_DATA_PATH = join(PROJECT_FILE_PATH, 'data')
DATA_PATH = setup.LOCAL_DATA_PATH
EXTERNAL_STORAGE_PATH = setup.LOCAL_EXTERNAL_STORAGE_PATH

OUT_DATA_PATH = join(PROJECT_FILE_PATH, "outdata")
EXTERNAL_STORAGE_OUT_PATH = join(EXTERNAL_STORAGE_PATH, 'outdata')

QOL_FILE_PATH = join(PROJECT_FILE_PATH, join("data", "qol"))
QOL_FILE_NAME = join(QOL_FILE_PATH, "ric_qol_{months}mnd.xlsx")

MAIN_DOSE_DIR_NAME = join("DPL", "0_70")
BOOST_DIR_NAME = join("DPL", "70_76")

MLE_PLOT_PATH = join(OUT_DATA_PATH, "likelihood")
LSQ_PLOT_PATH = join(OUT_DATA_PATH, "leastsquares")

DVH_COMPARISON_PATH = join(OUT_DATA_PATH, "dvh-analysis")
DVH_ARMS_COMPARISON_NAME = "dvh_arms-{structure}-{time}.png"
DVH_OUTCOME_COMPARISON_NAME = "dvh_outcome-arm_{study_arm}-{structure}-" \
                              "{outcome}-threshold_{threshold}-{time}.png"
DVH_REGISTRATION_COMPARISON_NAME = "dvh_registration-{structure}-{time}.png"

PCA_PATH = join(OUT_DATA_PATH, "pca")


# Read/write contour points
POINTS_FILE_PATH = join(PROJECT_DATA_PATH, "contours")
POINTS_FILE_NAME = "points-{roi_name}-id{patient_id}.txt"


# ===========================================
#   IMAGE REGISTRATION FILE & PATH CONSTANTS
# ===========================================

# Image registration data - base path
IMAGEREG_PARAMETER_PATH = join(PROJECT_DATA_PATH, 'parameterfiles')

# Image registration outdata - base path
IMAGEREG_OUT_PATH = join(EXTERNAL_STORAGE_OUT_PATH, 'image-reg')   # Write imagereg files to hard drive
#IMAGEREG_OUT_PATH = join(OUT_DATA_PATH, 'image-reg')   # Write imagereg files to local disk
ELASTIX_OUT_DIR = join(IMAGEREG_OUT_PATH, 'elastix')
TRANSFORMIX_OUT_DIR = join(IMAGEREG_OUT_PATH, 'transformix')


IMAGE_FILENAME          = "image-id{patient_id}.nii"
STRUCTURE_FILENAME      = "{structure}-id{patient_id}.nii"
TRANSFORMED_STRUCTURE_FILENAME = "{structure}-id{patient_id}-transformed.nii"
DOSE_FILENAME           = "dose-id{patient_id}.nii"
TRANSFORMED_DOSE_FILENAME = "transformed_dose-id{patient_id}.nii"
DEFORMATION_FILENAME    = "deformation-id{patient_id}.nii"
RESULT_IMAGE_FILENAME   = "result-id{patient_id}-{time}.nii"


RECTUM_DSC_ABOVE_0_5 = [3, 7, 19, 23, 36, 43, 47, 59, 60, 73, 80, 87,
                        100, 107, 111, 122, 128, 143, 148, 176, 184, 187, 190,
                        204, 207, 209, 218, 229, 237, 238, 239, 241]

# N=48 patients with DSC≥0.45: ID 196 & 227 (excluded, visual inspection)
RECTUM_DSC_ABOVE_0_45 = [1, 3, 7, 14, 19, 23, 36, 38, 43, 47, 52, 54, 59, 60,
                         73, 80, 87, 88, 98, 100, 107, 111, 122, 127, 128, 132,
                         143, 148, 153, 176, 179, 184, 185, 187, 190, 191, 203,
                         204, 207, 209, 218, 229, 234, 237, 238, 239, 241, 252]


# ===========================
#   DVH/DOSE FILE CONSTANTS
# ===========================

DVH_FILE_PATH = join(OUT_DATA_PATH, "DVH")
DVH_FILE_NAME = "dvh_id{patientid}_{organ}.txt"
DOSE_VOXEL_PATH = join(OUT_DATA_PATH, "voxeldata")
DOSE_VOXEL_FILE_NAME = "voxels_id{patientid}_{organ}.txt"

DELIMITER = '\t'
DVH_NUM_DECIMALS = 3


# ===========================
#   PLOTTING CONSTANTS
# ===========================
CT_MIN_INTENSITY = -150 # 'abdomen window'
CT_MAX_INTENSITY = 250  # 'abdomen window'
DOSE_MIN = 0
DOSE_MAX = 80
BASE_DOSE = 70
BOOST_DOSE = 78

ORANGE_RGB = (float(221)/255, float(97)/255, float(45)/255)
BLUE_RGB = (float(4)/255, float(110)/255, float(185)/255)

PLOT_PATH = join(OUT_DATA_PATH, "plots")
NTCP_PATH = join(OUT_DATA_PATH, "ntcp")
NTCP_LKB_PLOT_FILE_NAME = "ntcp_lkb_{organ}_{time}.png"
NTCP_LKB_TEXT_FILE_NAME = "ntcp_lkb_{organ}_{time}.txt"

BOXPLOT_FILE_NAME = "boxplot_dvh_{time}.png"


# ===========================
#   DICOM UID PREFIXES
# ===========================
DICOM_UID_PREFIX = '1.2.826.0.1.3680043.8.1070.'
DICOM_UID_PREFIX_IMAGE = DICOM_UID_PREFIX + '1.'
DICOM_UID_PREFIX_RTSTRUCT = DICOM_UID_PREFIX + '2.'
DICOM_UID_PREFIX_RTPLAN = DICOM_UID_PREFIX + '3.'
DICOM_UID_PREFIX_RTDOSE = DICOM_UID_PREFIX + '4.'