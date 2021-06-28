"""
Created on 08/04/2021
@author: ingridtveten

TODO: Description...
"""

import os
import SimpleITK as sitk
import numpy as np



# ===========================
#  SimpleITK FEATURES
# ===========================

def get_image_series_reader():

    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader


def get_danielsson_distance_map_filter():

    # edm = Euclidean distance map
    edm_filter = sitk.DanielssonDistanceMapImageFilter()
    edm_filter.InputIsBinaryOn()
    edm_filter.UseImageSpacingOn()
    edm_filter.SquaredDistanceOn()

    return edm_filter


# ===========================
#  INITIALIZE IMAGE SERIES
# ===========================

def get_patient_image_series_path(patient_id):

    from rtfilereader.dicomloader import get_patient_folder
    from constants import MAIN_DOSE_DIR_NAME

    patient_path = get_patient_folder(patient_id)
    full_path = os.path.join(patient_path, MAIN_DOSE_DIR_NAME)
    return full_path


def read_image_series(patient_id, reader=None, image_prefix='CT'):
    """

    Parameters
    ----------
    patient_id : int/str
        Patient ID for which to read images and generate 3D image
    reader : SimpleITK.ImageSeriesReader

    image_prefix : str (defaults to 'CT')


    Returns
    -------
    img : SimpleITK.Image
        A 3D image for the patient
    """

    patient_path = get_patient_image_series_path(patient_id)

    # Find right DICOM series (correct prefix in filename)
    dcm_series = reader.GetGDCMSeriesIDs(patient_path)
    image_files = None
    use_series_id = None
    for series_id in dcm_series:
        image_files = reader.GetGDCMSeriesFileNames(patient_path, series_id)
        # Check if a file in series has correct prefix. If so, use series ID
        fn = os.path.split(image_files[0])[1] # Get filename
        if fn.startswith(image_prefix):
            use_series_id = series_id
            break

    if image_files is None or use_series_id is None:
        msg = f"Error reading DICOM series for patient with ID {patient_id}."
        raise FileNotFoundError(msg)

    # Get files with correct filename
    dcm_files = reader.GetGDCMSeriesFileNames(patient_path, use_series_id)
    img_files = []
    for fp in dcm_files:
        fn = os.path.split(fp)[1] # Get tail of path = filename
        if fn.startswith(image_prefix) and fn.endswith('.dcm'):
            img_files.append(fp)

    # Read DICOM series
    if reader is None:
        reader = get_image_series_reader()
    reader.SetFileNames(img_files)
    image = reader.Execute()

    return image


# ===========================
#  CONTOUR MASKS
# ===========================

def generate_contour_masks(patient, ct_image, structures, write_to_file=True):
    """
    For the structures given by the list og StructureEnum, generate a contour
    mask with the same dimensions as the CT image.

    Parameters
    ----------
    patient : Patient

    ct_image : SimpleITK.Image

    structures : list of StructureEnum

    write_to_file : (optional) bool
        If True, the mask is written to file

    Returns
    -------
    contour_masks : dictionary of SimpleITK.Image
        Dictionary of contour masks (SimpleITK.Image), indexed by StructureEnum
    """

    import constants
    from utilities.util import find_roi_number_from_roi_name
    import imageregistration.imageprocessing as processing

    patient_id = patient.get_id()

    contour_masks = {}
    for struct in structures:
        roi_id = find_roi_number_from_roi_name(patient, struct)
        roi = patient.get_roi(roi_id)
        roi_mask = processing.get_contour_ct_mask(ct_image, roi)

        roi_mask, dxdydz = processing.image_centre_to_origin(roi_mask)

        contour_masks[struct] = roi_mask
        if write_to_file is not None:
            filename = constants.STRUCTURE_FILENAME.format(structure=struct.name,
                                                     patient_id=patient_id)
            write_image_to_file(roi_mask, filename)

    return contour_masks


def combine_contour_masks(contour_masks):
    """
    Combine the masks (union) in the dictionary

    Parameters
    ----------
    contour_masks : dictionary of SimpleITK.Image
        Dictionary of contour masks (SimpleITK.Image), indexed by StructureEnum

    Returns
    -------
    combined_mask : SimpleITK.Image
        The union of the masks in the contour_masks dictionary
    """

    if not len(list(contour_masks.keys())):
        print("No masks are present in the dictionary passed to "
              "'combine_contour_masks'.")

    masks = list(contour_masks.values())
    combined_mask = masks[0]
    if len(masks) > 1:
        for m in masks:
            combined_mask = sitk.Or(combined_mask, m)
    return combined_mask


def get_structure_probability_distribution(patients, structure,
                                           plot_title=None, path=None):

    import utilities.util as util
    import matplotlib.pyplot as plt

    masks = []
    for i in patients:
        try:
            m = util.get_mask_from_file(structure, i)
            masks.append(sitk.GetArrayFromImage(m))
        except:
            try:
                m = util.get_mask_from_file(
                    structure, i, is_fixed_patient=True)
                fixed_mask = sitk.GetArrayFromImage(m)
                masks.append(fixed_mask)
            except:
                print(f"Could not find mask for patient ID {i}")

    dx, dy, dz = m.GetSpacing()
    Nx, Ny, Nz = m.GetSize()

    masks = np.asarray(masks).astype(dtype=int)
    probability = np.average(masks, axis=0)
    probability_plane = probability[:, :, Nx//2]
    fixed_mask_plane = fixed_mask[:, :, Nx//2]

    FS_CBAR_LABEL = 18  # colorbar label fontsize
    FS_AXTICKS = 14     # tick marks fontsize

    PROB_KW = {'cmap': 'hot', 'vmin': 0, 'vmax': 1, 'alpha': 1.0}
    MASK_KW = {'colors': 'blue', 'levels': [1.0],  'linewidths': 2}
    CBAR_KW = {'orientation': 'vertical', 'shrink': 1.0}

    fig, ax = plt.subplots(figsize=(7, 6))
    #if plot_title is not None:
    #    fig.suptitle(plot_title, fontsize=16)

    prob = ax.imshow(probability_plane, **PROB_KW)
    ax.contour(fixed_mask_plane, **MASK_KW)  # Plot contour mask as overlay

    cb = fig.colorbar(prob, ax=ax, **CBAR_KW)
    cb.ax.set_ylabel('Probability', fontsize=FS_CBAR_LABEL)
    cb.ax.tick_params(labelsize=FS_AXTICKS)

    ax.set_aspect(aspect=float(dz) / dy)
    ax.set_xlim(180, 350)
    ax.set_ylim(28, 100)
    ax.tick_params(axis="x", labelsize=FS_AXTICKS)
    ax.tick_params(axis="y", labelsize=FS_AXTICKS)

    if path is not None:
        t0 = util.timestamp()
        util.save_figure(fig, path=save_to,
                         fn=f"rectum-probability-N={len(patients)}-{t0}.png")
    fig.show()



# ===========================
#  WRITE IMAGE TO FILE
# ===========================

def write_image_to_file(image, filename):

    from constants import IMAGEREG_OUT_PATH

    IMAGEREG_OUT_PATH = os.path.join(IMAGEREG_OUT_PATH, 'compare_dsc')

    if not os.path.isdir(IMAGEREG_OUT_PATH):
        os.makedirs(IMAGEREG_OUT_PATH)

    image_writer = sitk.ImageFileWriter()
    filepath = os.path.join(IMAGEREG_OUT_PATH, filename)
    image_writer.SetFileName(filepath)
    image_writer.Execute(image)


# ===========================
#  PRINT FUNCTIONS
# ===========================

def print_parameter_map(parameter_map):
    sitk.PrintParameterMap(parameter_map)


def print_image_information(image):
    """
    Print some basic image information using SimpleITK commands

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image to print information for
    """

    print(f"\nImage information:"
          f"\nDimensions:       {image.GetDimension()}"
          f"\nImage size:       {image.GetSize()}"
          f"\nImage origin:     {image.GetOrigin()}"
          f"\nPixel spacing:    {image.GetSpacing()}"
          f"\nImage direction:  {image.GetDirection()}"
          f"\n"
          )


if __name__ == "__main__":
    import constants
    from utilities.util import filter_study_arms, StructureEnum

    ARM_B = filter_study_arms(constants.AVAILABLE_PATIENTS, 2)
    USED_FOR_VBA = [31] + constants.RECTUM_DSC_ABOVE_0_5
    DSC_ABOVE_0_4 = [31] + constants.RECTUM_DSC_ABOVE_0_45

    save_to = '/Users/ingridtveten/Documents/Skole/5-NTNU/2020-2021/0-Masteroppgave/Figurer/4-Results/Rectum-probability'

    get_structure_probability_distribution(
        ARM_B, StructureEnum.RECTUM,
        plot_title="Probability of rectum per voxel (arm B)", path=save_to)
    get_structure_probability_distribution(
        USED_FOR_VBA, StructureEnum.RECTUM,
        plot_title="Probability of rectum per voxel (rectum DSC > 0.5)",
        path=save_to)
    '''get_structure_probability_distribution(
            DSC_ABOVE_0_4, StructureEnum.RECTUM,
            plot_title="Probability of rectum per voxel (rectum DSC > 0.4)",
            path=save_to)'''