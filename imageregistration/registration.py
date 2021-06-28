"""
Created on 26/03/2021
@author: ingridtveten

TODO: Description...
"""

import os
import logging

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

logger = logging.getLogger()


def ct_to_ct_registration():

    import constants

    import utilities.util as util
    from utilities.util import StructureEnum
    import imageregistration.util as imageutil
    import imageregistration.imageprocessing as processing
    import imageregistration.plotting as imregplot


    # =========== SETTINGS ===========
    ARM_B = util.filter_study_arms(constants.AVAILABLE_PATIENTS, wanted_arm=2)

    WRITE_INITIAL_DCM_AS_NIFTI = True
    WRITE_RESULT_AS_NIFTI = True
    T_START = util.timestamp()

    # =========== CONSTANTS ===========
    FIXED_ID = 31
    STRUCTURES = [StructureEnum.RECTUM, StructureEnum.BLADDER,
                  StructureEnum.PTV_0_70]


    # ====================================================
    #       SimpleITK/Elastix/Transformix components
    # ====================================================
    logger.info(f"{util.timestamp()}\t"
                f"Getting SimpleITK components")

    # ImageSeriesReader
    reader = imageutil.get_image_series_reader()

    # ElastixImageFilter
    elastixFilter = get_elastix_image_filter()

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(get_parameter_map('affine-parameters.txt'))
    #parameterMapVector.append(get_parameter_map('bspline-parameters.txt'))
    elastixFilter.SetParameterMap(parameterMapVector)

    #elastixFilter.SetParameterMap(get_default_parameter_map())
    #elastixFilter.SetParameter("MaximumNumberOfIterations", "512")

    # TransfromixImageFilter
    transformixFilter = get_transformix_image_filter()

    # Similarity filters
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_filter = sitk.HausdorffDistanceImageFilter()


    # ====================================================
    #       Fixed image information
    # ====================================================
    logger.info(f"{util.timestamp()}\t"
                    f"Getting fixed image information for ID {FIXED_ID}")

    # Read Patient, DoseMatrix, CT image series
    fix_patient = util.initialize_patient(patient_id=FIXED_ID)
    fix_dose    = fix_patient.get_dose_matrix()
    fixed_image = imageutil.read_image_series(FIXED_ID, reader)

    # Fixed image mask to focus registration
    fixed_contours = imageutil.generate_contour_masks(
        fix_patient, fixed_image, STRUCTURES,
        write_to_file=True
    )

    #  Get union of rectum, bladder, PTV as fixed mask
    fixed_mask = imageutil.combine_contour_masks(fixed_contours)
    # Dilate mask
    R = 10 # Dilation radius in image coords, mm
    fixed_mask = sitk.BinaryDilate(fixed_mask, (R, R, R))
    imageutil.write_image_to_file(fixed_mask,
                                  f"mask-id{FIXED_ID}-dilate{R}.nii")

    # Preprocessing
    fixed_image, fix_dxdydz = processing.image_centre_to_origin(fixed_image)
    # NOTE: DoseMatrix dimensions may differ from CT image dimensions, so
    #   cannot necessarily move image centre to origin as with image and mask
    fix_dose.translate(fix_dxdydz)
    fix_dose_itk = processing.dose_matrix_to_SimpleITK_image(fix_dose)

    # Resample fixed dose image to same dimensions as CT image
    resampled_dose = sitk.Resample(fix_dose_itk,     # image to resample
                                   fixed_image,      # reference image
                                   sitk.Transform(), # transform
                                   sitk.sitkBSpline, # interpolator, 3rd order
                                   0,   # default value
                                   fix_dose_itk.GetPixelID() # pixel type
                                   )

    if WRITE_INITIAL_DCM_AS_NIFTI:
        # Fixed CT image
        image_filename = constants.IMAGE_FILENAME.format(patient_id=FIXED_ID)
        imageutil.write_image_to_file(fixed_image, image_filename)
        # Dose distribution
        dose_filename = constants.DOSE_FILENAME.format(patient_id=FIXED_ID)
        imageutil.write_image_to_file(resampled_dose, dose_filename)


    elastixFilter.SetFixedImage(fixed_image) # SET FIXED IMAGE
    #elastixFilter.AddFixedImage(fix_normalizedRectumDistMap)
    elastixFilter.SetFixedMask(fixed_mask)



    # ====================================================
    #       LOOP THROUGH MOVING IMAGES HERE!
    # ====================================================
    logger.info(f"{util.timestamp()}\t"
                    f"Entering registration loop")

    print(f"{'ID':<3}\t{'Structure':<12}\t{'DSC':<8}\t{'HD':<8}")
    for patient_id in ARM_B:

        if patient_id != FIXED_ID and patient_id < 5:
            MOVING_ID = patient_id
        else: continue

        T_CURR_REG = util.timestamp()


        # ====================================================
        #       Moving image information
        # ====================================================
        logger.info(f"{util.timestamp()}\t"
                    f"Getting moving image information for ID {MOVING_ID}")

        # Read Patient, DoseMatrix, CT image series
        mov_patient  = util.initialize_patient(patient_id=MOVING_ID)
        mov_dose     = mov_patient.get_dose_matrix()
        moving_image = imageutil.read_image_series(MOVING_ID, reader)

        # Moving image contours
        moving_contours = imageutil.generate_contour_masks(
            mov_patient, moving_image, STRUCTURES,
            write_to_file=True
        )

        # CT image preprocessing
        moving_image, dxdydz = processing.image_centre_to_origin(moving_image)

        # Dose preprocessing
        mov_dose.translate(dxdydz)
        mov_dose_itk = processing.dose_matrix_to_SimpleITK_image(mov_dose)

        if WRITE_INITIAL_DCM_AS_NIFTI:
            # Moving CT image
            image_filename = constants.IMAGE_FILENAME.format(
                patient_id=MOVING_ID)
            imageutil.write_image_to_file(moving_image, image_filename)
            # Dose distribution
            dose_filename = constants.DOSE_FILENAME.format(patient_id=MOVING_ID)
            imageutil.write_image_to_file(mov_dose_itk, dose_filename)

    
        elastixFilter.SetMovingImage(moving_image)


        # ====================================================
        #       Register 3D images and get result
        # ====================================================
        logger.info(f"{util.timestamp()}\t"
                    f"Registering ID {MOVING_ID} to {FIXED_ID}")

        elastixFilter.Execute()
    
    
        result_image = elastixFilter.GetResultImage()
        if WRITE_RESULT_AS_NIFTI:
            result_image = sitk.Cast(result_image, sitk.sitkInt32)
            image_filename = constants.RESULT_IMAGE_FILENAME.format(
                patient_id=MOVING_ID, time=T_CURR_REG)
            imageutil.write_image_to_file(result_image, image_filename)


        # ====================================================
        #       Transform CT image, contour points, dose
        # ====================================================
        logger.info(f"{util.timestamp()}\t"
                    f"Applying transform to moving image, dose, contours")

        transformParamMapVector = elastixFilter.GetTransformParameterMap()

        # ====================================================
        #       Get deformation field as Nifti
        # ====================================================
        logger.info(f"{util.timestamp()}\t"
                    f"Computing result image and deformation field")

        transformixFilter.SetTransformParameterMap(transformParamMapVector)

        transformixFilter.ComputeDeformationFieldOn()
        transformixFilter.SetMovingImage(moving_image) # Needs dummy image
        transformixFilter.Execute()

        deformation_field = transformixFilter.GetDeformationField()
        image_filename = constants.DEFORMATION_FILENAME.format(
            patient_id=MOVING_ID)
        imageutil.write_image_to_file(deformation_field, image_filename)


        # ====================================================
        #       Transform contour mask
        # ====================================================
        transformixFilter.ComputeDeformationFieldOff()

        # NOTE: FinalBSplineInterpolationOrder = 0 to get binary result
        transformixFilter.SetTransformParameter(
            "FinalBSplineInterpolationOrder", "0")

        # Moving image contours
        for struct in STRUCTURES:

            logger.info(f"{util.timestamp()}\t"
                    f"Transforming {struct.value} for ID {MOVING_ID}")

            mask = moving_contours[struct]
            transformixFilter.SetMovingImage(mask)
            transformixFilter.Execute()

            warped_mask = transformixFilter.GetResultImage()
            image_filename = constants.TRANSFORMED_STRUCTURE_FILENAME.format(
                patient_id=FIXED_ID, structure=struct.name)
            imageutil.write_image_to_file(warped_mask, image_filename)

            # Similarity measures
            mask_as_int = sitk.Cast(warped_mask, sitk.sitkUInt8)

            dice_filter.Execute(fixed_contours[struct], mask_as_int)
            hausdorff_filter.Execute(fixed_contours[struct], mask_as_int)

            dsc = dice_filter.GetDiceCoefficient()
            hd = hausdorff_filter.GetHausdorffDistance()
            print(f"{patient_id:<3}\t{struct.value:<12}\t{dsc:<8.4f}\t{hd:<8.4f}")

        transformixFilter.SetTransformParameter(
            "FinalBSplineInterpolationOrder", "3")

        # ====================================================
        #       Transform dose matrix
        # ====================================================
        logger.info(f"{util.timestamp()}\t"
                    f"Transforming dose distribution")

        transformixFilter.SetMovingImage(mov_dose_itk)
        transformixFilter.Execute()

        transformed_dose_image = transformixFilter.GetResultImage()
        image_filename = constants.TRANSFORMED_DOSE_FILENAME.format(
            patient_id=MOVING_ID)
        imageutil.write_image_to_file(transformed_dose_image, image_filename)

        '''transformed_dose_arr = sitk.GetArrayFromImage(transformed_dose_image)
        imregplot.plot_3d_image_series(
            transformed_dose_arr, grid_edge_size=3,
            plot_title=f"Transformed dose ID {MOVING_ID} to {FIXED_ID} coords",
            cmap='jet', save_fig=False)'''





# ========================================
#   Support functions
# ========================================

def get_elastix_image_filter():
    """

    Returns
    -------
    elastixImageFilter : sitk.ElastixImageFilter
    """

    from constants import ELASTIX_OUT_DIR

    if not os.path.isdir(ELASTIX_OUT_DIR):
        os.makedirs(ELASTIX_OUT_DIR)

    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.SetOutputDirectory(ELASTIX_OUT_DIR)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOn()

    return elastixImageFilter


def get_transformix_image_filter():
    """

    Returns
    -------
    transformixImageFilter : sitk.TransformixImageFilter
    """

    from constants import TRANSFORMIX_OUT_DIR

    if not os.path.isdir(TRANSFORMIX_OUT_DIR):
        os.makedirs(TRANSFORMIX_OUT_DIR)

    transformixImageFilter = sitk.TransformixImageFilter()

    transformixImageFilter.SetOutputDirectory(TRANSFORMIX_OUT_DIR)
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.LogToFileOn()

    transformixImageFilter.ComputeSpatialJacobianOff()
    transformixImageFilter.ComputeDeterminantOfSpatialJacobianOff()
    transformixImageFilter.ComputeDeformationFieldOn()

    return transformixImageFilter


def get_default_parameter_map():
    elx = sitk.ElastixImageFilter()
    return elx.GetParameterMap()


def get_parameter_map(parameter_filename):
    from constants import IMAGEREG_PARAMETER_PATH

    # Multi-metric registration
    default_map = sitk.GetDefaultParameterMap('affine')
    default_map['Registration'] = ['MultiMetricMultiResolutionRegistration']
    original_metric = default_map['Metric']
    default_map['Metric'] = [original_metric[0],
                             'CorrespondingPointsEuclideanDistanceMetric']
    #parameter_map = default_map

    parameter_path = os.path.join(IMAGEREG_PARAMETER_PATH, parameter_filename)
    parameter_map = sitk.ReadParameterFile(parameter_path)
    return parameter_map


def get_contour_points_full_path(patient_id, structure):
    from constants import POINTS_FILE_PATH, POINTS_FILE_NAME

    contour_path = os.path.join(POINTS_FILE_PATH, structure.name)
    filename = POINTS_FILE_NAME.format(roi_name=structure.name,
                                       patient_id=patient_id)
    return os.path.join(contour_path, filename)




if __name__ == "__main__":
    from time import time
    from utilities.logger import init_logger, close_logger
    t0 = time()

    init_logger('ntcp-model')
    logger.setLevel(logging.WARNING)


    ct_to_ct_registration()


    t = time()
    time_str = f"Process took {t - t0:.6f} s"
    print(f"\n\n--------------------------\n{time_str}")
    close_logger()