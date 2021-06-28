"""
Created on 30/09/2020
@author: ingridtveten
"""

import logging
import numpy as np
from enum import Enum

from constants import AVAILABLE_PATIENTS, NO_PATIENT_DATA


logger = logging.getLogger()


# =====================================
#   ENUM TYPES
# =====================================

class DICOMModality(Enum):
    """ Enum class for DICOM modalities """
    CT = "CT"
    MR = "MR"
    RTDOSE = "RTDOSE"
    RTIMAGE = "RTIMAGE"
    RTPLAN = "RTPLAN"
    RTSTRUCT = "RTSTRUCT"


class DoseGroup(Enum):
    """ Enum class for mapping dose groups and directory names """
    BASE = "0_70"
    BOOST = "70_76"
    TOTAL = "Total"


# =====================================
#   STRUCTURE MAPPING/HELPER FUNCTIONS
# =====================================

class StructureEnum(Enum):
    """ Enum class for mapping ROI structures and their names """
    CTV_0_70 = "CTV2 0-70"
    CTV_70_78 = "CTV1 70-78"
    PTV_0_70 = "PTV2 0-70"
    PTV_70_78 = "PTV1 70-78"
    RECTUM = "Rectum"
    RECTAL_MUCOSA = "Rectal mucosa"
    ANAL_CANAL = "Anal canal"
    PENILE_BULB = "Penile bulb"
    TESTICLE_SIN = "Left testicle"
    TESTICLE_DX = "Right testicle"
    BLADDER = "Bladder"
    RECTAL_WALL = "Rectal wall"
    VOID = ""


def map_name_to_structure_enum(roi_name):
    """
    Searches given ROI name for certain substrings to identify which type of
        ROI it is and which StructureEnum attribute to assign.

    Parameters:
    -----------
    roi_name : str
        Name of a ROI that should be mapped to a StructureEnum type
    """

    lowercase = roi_name.lower()

    # ==== CTV & PTV ====
    if "ptv" in lowercase:
        if "ptv2" in lowercase: return StructureEnum.PTV_0_70
        if "ptv1" in lowercase: return StructureEnum.PTV_70_78
        if "ptv 2" in lowercase: return StructureEnum.PTV_0_70
        if "ptv 1" in lowercase: return StructureEnum.PTV_70_78
        if "70" in lowercase and "78" in lowercase:
            return StructureEnum.PTV_70_78
        if "0" in lowercase and "70" in lowercase:
            return StructureEnum.PTV_0_70
    if "ctv" in lowercase:
        if "ctv2" in lowercase: return StructureEnum.CTV_0_70
        if "ctv1" in lowercase: return StructureEnum.CTV_70_78
        if "ctv 2" in lowercase: return StructureEnum.CTV_0_70
        if "ctv 1" in lowercase: return StructureEnum.CTV_70_78
        if "70" in lowercase and "78" in lowercase:
            return StructureEnum.CTV_70_78
        if "0" in lowercase and "70" in lowercase:
            return StructureEnum.CTV_0_70

    # ==== Rectum and rectal mucosa ====
    if ("mucosa" in lowercase):
        return StructureEnum.RECTAL_MUCOSA

    if ("rect" in lowercase):
        if ("mucosa" in lowercase):
            return StructureEnum.RECTAL_MUCOSA
        else:#if ("rectum" == lowercase):
            return StructureEnum.RECTUM
    if ("rekt" in lowercase):
        if ("mucosa" in lowercase):
            return StructureEnum.RECTAL_MUCOSA
        elif ("slimhinne" in lowercase):
            return StructureEnum.RECTAL_MUCOSA
        else:#if ("rektum" == lowercase):
            return StructureEnum.RECTUM

    # ==== Penile bulb ====
    if ("penile" in lowercase or "bulb" in lowercase):
        return StructureEnum.PENILE_BULB

    # ==== Testicles ====
    if ("tes" in lowercase or "tstsis" in lowercase or "tetis" in lowercase) \
            and ("sin" in lowercase or "v" in lowercase):
        return StructureEnum.TESTICLE_SIN
    if ("tes" in lowercase or "tstsis" in lowercase or "tetis" in lowercase) \
            and ("dex" in lowercase or "dx" in lowercase or "h" in lowercase):
        return StructureEnum.TESTICLE_DX

    # ==== Anal canal ====
    if "anal" in lowercase:
        return StructureEnum.ANAL_CANAL
    if "anus" in lowercase:
        return StructureEnum.ANAL_CANAL

    # ==== Bladder ====
    if ("bl" in lowercase or "vesica" in lowercase or "urin" in lowercase):
        return StructureEnum.BLADDER

    logger.debug(f"Could not match structure name {roi_name} to StructureEnum")
    return StructureEnum.VOID


def find_roi_number_from_roi_name(patient, structure_of_interest):
    """
    Find ROI number by looping through ROI names and matching the name to
        StructureEnum attribute

    Parameters:
    -----------
    patient : Patient
        The patient for which the ROI number should be found
    structure_of_interest : StructureEnum
        The ROI for which the ROI number should be found

    Raises:
    -------
    ROIError : Raises ROIError if no ROI names can be matched to StructureEnum
    """
    from utilities.exceptions import ROIError

    structures = patient.get_rt_structures()

    found_roi = False
    for roi_id in structures.keys():
        roi_name = patient.get_roi_name(roi_id)
        structure = map_name_to_structure_enum(roi_name)
        if structure == structure_of_interest:
            #found_roi = True
            return roi_id

    if not found_roi:
        msg = "Did not find ROI matching '{}' for patient {}".format(
            structure_of_interest.value, patient.patient_id)
        raise ROIError(msg)


#================================================================
#   PATIENT HELPER FUNCTIONS
#================================================================

def initialize_patients(from_id=1, to_id=260):
    """
    Initializes a dictionary of all patients in the specified range.

    Warning
    -------
    Requires significant amounts of memory to store all patient data (images,
        dose matrix, structures), so function is not used when running on my
        laptop/personal computer.

    Returns
    -------
    patients : dictionary
        Dictionary of Patient objects indexed by patient ID
    """

    patients = {}
    for patient_id in range(from_id, to_id):
        pt = initialize_patient(patient_id)
        if pt is not None:
            patients[patient_id] = pt
    return patients


def initialize_patient(patient_id):
    """
    Initializes patient with the specified id.

    Returns
    -------
    patient : Patient
        Patient object with the given ID
    """

    from patient import Patient

    if patient_id in NO_PATIENT_DATA: return None

    try:
        patient = Patient(patient_id)
        #print("{}\tFound patient".format(patient_id))
        return patient
    except FileNotFoundError as err:
        print("{}\tNo patient with ID {}\t{}".format(patient_id, patient_id, err))
    except Exception as err:
        print("{}\tError for patient ID {}\t{}".format(patient_id, patient_id, err))

    return None


def print_structures(id=None):
    """
    Helper function. Prints all ROI IDs and names for the patients specified by
        the 'id' argument.

    Parameters
    ----------
    id : int / tuple of ints
        Specifies which patient(s) for which to print ROI IDs and names
    """
    from patient import Patient

    if isinstance(id, int):
        patient = Patient(id)
        patient.print_roi_id_and_names()
    elif isinstance(id, tuple):
        for i in id:
            patient = Patient(i)
            patient.print_roi_id_and_names()
            print()
    else:
        print("Specify the argument 'id' as integer or tuple of integers")


def print_structure_names(struct, id=(1, 10)):
    """
    Helper function. Prints ROI names for the specified structure and for all
        patients in the range specified by the 'id' argument.
    """
    from utilities.util import find_roi_number_from_roi_name

    print("Printing names of * {} * for patients {} to {}".format(
        struct.value, id[0], id[1]))

    for id in range(id[0], id[1]+1):
        try:
            patient = initialize_patient(id)
            roi_id = find_roi_number_from_roi_name(patient, struct)
            print("{}\t{}".format(id, patient.get_roi_name(roi_id)))
        except Exception as err:
            print("{}\tNo available ROI".format(id, ))


# =====================================
#   IMAGE PROCESSING HELPER FUNCTIONS
# =====================================

def get_scaled_image(img_data):
    """
    To convert the stored data to HU it needs to be scaled with information
    provided in the DICOM header
    """
    array = img_data.pixel_array.astype('float')  # convert from uint to float
    slope = img_data.RescaleSlope
    intercept = img_data.RescaleIntercept

    new_array = array * slope + intercept
    return new_array


def patient_to_image_coordinates(points, ct):
    """
    RTSTRUCT file are in patient/scanner coord. Function converts to image/pixel
        coords and ensures contour is closed by repeating the first point.
    """
    IPP = ct['ImagePositionPatient']
    spacing = ct['PixelSpacing']
    new_pts = {'x': [(value - IPP[0]) / spacing[0] for value in points['x']],
               'y': [(value - IPP[1]) / spacing[1] for value in points['y']] }
    # Make sure contour is 'closed'
    new_pts['x'].append(new_pts['x'][0])
    new_pts['y'].append(new_pts['y'][0])

    return new_pts


def find_centroid_2d(points):
    """
    Find the centroid of the structure defined by a list of 2D points.

    Parameters
    ----------
    points : ndarray, array-like

    """

    N = len(points)
    centroid = [0, 0]
    signed_area = 0

    # Iterate through points (vertices)
    for i in range(N):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % N]
        # Calculate value of A using shoelace formula
        A = (x0 * y1) - (x1 * y0)
        signed_area += A
        # Calculate coordinates of centroid of polygon
        centroid[0] += (x0 + x1) * A
        centroid[1] += (y0 + y1) * A

    signed_area *= 0.5
    centroid[0] = (centroid[0]) / (6 * signed_area)
    centroid[1] = (centroid[1]) / (6 * signed_area)
    return centroid


def cartesian_to_polar(x, y):
    """
    Get polar coordinates (rho, `theta`) from cartesian coordinates (x, y). The
    angle `theta` is  given in radians.
    """
    rho = np.sqrt(np.square(x) + np.square(y))
    theta = np.arctan2(y, x)
    return rho, theta


def polar_to_cartesian(rho, theta):
    """
    Get cartesian coordinates (x, y) from polar coordinates (rho, `theta`). The
    angle `theta` is  given in radians.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


# =====================================
#   FILE PATH HELPER FUNCTIONS
# =====================================

def list_dicom_files_in_dir(dir_path):
    """ Lists all DICOM file paths in the given directory """
    from os import listdir
    from os.path import isfile, join

    path_list = []

    for f in listdir(dir_path):
        filepath = join(dir_path, f)
        if isfile(filepath) and f.endswith(".dcm"):
            path_list.append(filepath)

    return path_list


def list_dicom_files_in_dir_full(dir_path):
    """ Lists all DICOM file paths in given directory and all subdirectories """
    from os import walk
    from os.path import isfile, join

    path_list = []

    for (dirpath, dirnames, filenames) in walk(dir_path):
        for fn in filenames:
            filepath = join(dirpath, fn)
            if isfile(filepath) and fn.endswith(".dcm"):
                path_list.append(filepath)

    return path_list


def save_figure(fig, path, fn):
    from os import makedirs
    from os.path import join, isdir
    if not isdir(path):
        makedirs(path)
    fig.savefig(join(path, fn), format='png')


def save_lkb_params(td50, td50_std, n, n_std, m, m_std,
                    num_samples, description, confidence_level, path, fn):
    from scipy import stats
    from os import makedirs
    from os.path import join, isdir

    txt = description
    txt += "\nParameter\tValue\tSTD"
    txt += "\n{}\t{:.3f}\t{:.3f}".format("TD50", td50, td50_std)
    txt += "\n{}\t{:.3f}\t{:.3f}".format("n", n, n_std)
    txt += "\n{}\t{:.3f}\t{:.3f}".format("m", m, m_std)
    txt += "\n"

    conf_int_td50 = stats.norm.interval(
        confidence_level, loc=td50, scale=td50_std / np.sqrt(num_samples))
    conf_int_n = stats.norm.interval(
        confidence_level, loc=n, scale=n_std / np.sqrt(num_samples))
    conf_int_m = stats.norm.interval(
        confidence_level, loc=m, scale=m_std / np.sqrt(num_samples))
    txt += "\n{:.0f} % confidence intervals".format(100 * confidence_level)
    txt += "\nTD50: ({:.3f} - {:.3f})".format(conf_int_td50[0], conf_int_td50[1])
    txt += "\nn: ({:.3f} - {:.3f})".format(conf_int_n[0], conf_int_n[1])
    txt += "\nm: ({:.3f} - {:.3f})".format(conf_int_m[0], conf_int_m[1])

    if not isdir(path):
        makedirs(path)
    f = open(join(path, fn), 'w')
    f.write(txt)
    f.close()


# =====================================
#   PRINT/LOG FUNCTIONS
# =====================================

def print_dvh_stats(structure_dict, dvh_dict):
    """ Print dose/volume statistics for the structures and DVH passed """
    # NOTE: DVH.volume returns DVH.differential.counts.sum()
    print("{:<17}\t{:<15}\t{:<10}\t{:<10}\t{:<10}"
        "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
        "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
        "Structure", "Volume [cm3]", "D_min [Gy]", "D_mean [Gy]", "D_max [Gy]",
        "D2cc [Gy]", "D2 [Gy]", "D50 [Gy]", "D60 [Gy]", "D70 [Gy]", "D90 [Gy]",
            "D95 [Gy]", "D98 [Gy]",
        "V50 [%]", "V60 [%]", "V65 [%]", "V70 [%]", "V75 [%]", "V80 [%]"))
    #print("{:-<150}".format(""))
    for key, structure in structure_dict.items():
        if (key in dvh_dict):
            dvh = dvh_dict[key]
            rel_dvh = dvh.relative_volume
            print("{:<17}\t{:<15.1f}\t{:<10.2f}\t{:<10.2f}\t{:<10.2f}"
                "\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}"
                    "\t{:<8.2f}\t{:<8.2f}"
                "\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}"
                    "\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}".format(
                structure.name, dvh.volume, dvh.min, dvh.mean, dvh.max,
                dvh.D2cc.value, rel_dvh.D2.value, rel_dvh.D50.value,
                    rel_dvh.D60.value, rel_dvh.D70.value, rel_dvh.D90.value,
                    rel_dvh.D95.value, rel_dvh.D98.value,
                rel_dvh.statistic('V50Gy').value, rel_dvh.statistic('V60Gy').value,
                rel_dvh.statistic('V65Gy').value, rel_dvh.statistic('V70Gy').value,
                rel_dvh.statistic('V75Gy').value, rel_dvh.statistic('V80Gy').value))

    log_dvh_stats(structure_dict, dvh_dict)

def log_dvh_stats(structure_dict, dvh_dict):
    """ Write dose/volume statistics for the structures and DVH to logfile """
    # NOTE: DVH.volume returns DVH.differential.counts.sum()
    logger.debug("{:<17}\t{:<15}\t{:<10}\t{:<10}\t{:<10}"
        "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
        "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
        "Structure", "Volume [cm3]", "D_min [Gy]", "D_mean [Gy]", "D_max [Gy]",
        "D2cc [Gy]", "D2 [Gy]", "D50 [Gy]", "D60 [Gy]", "D70 [Gy]", "D90 [Gy]",
            "D95 [Gy]", "D98 [Gy]",
        "V50 [%]", "V60 [%]", "V65 [%]", "V70 [%]", "V75 [%]", "V80 [%]"))
    for key, structure in structure_dict.items():
        if (key in dvh_dict):
            dvh = dvh_dict[key]
            rel_dvh = dvh.relative_volume
            logger.debug("{:<17}\t{:<15.1f}\t{:<10.2f}\t{:<10.2f}\t{:<10.2f}"
                "\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}"
                    "\t{:<8.2f}\t{:<8.2f}"
                "\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}"
                    "\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}".format(
                structure.name, dvh.volume, dvh.min, dvh.mean, dvh.max,
                dvh.D2cc.value, rel_dvh.D2.value, rel_dvh.D50.value,
                    rel_dvh.D60.value, rel_dvh.D70.value, rel_dvh.D90.value,
                    rel_dvh.D95.value, rel_dvh.D98.value,
                rel_dvh.statistic('V50Gy').value, rel_dvh.statistic('V60Gy').value,
                rel_dvh.statistic('V65Gy').value, rel_dvh.statistic('V70Gy').value,
                rel_dvh.statistic('V75Gy').value, rel_dvh.statistic('V80Gy').value))


def print_structure_stats_header(structure_enum):
    """ Print header for dose/volume statistics for structures """
    print_structure_name(structure_enum)
    print("{:<10}\t{:<16}\t{:<15}"
          "\t{:<10}\t{:<10}\t{:<10}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
        "Patient ID", "ROI Name", "Volume [cm3]",
        "D_min [Gy]", "D_mean [Gy]", "D_max [Gy]", "D2cc [Gy]",
        "D2 [Gy]", "D50 [Gy]", "D90 [Gy]", "D98 [Gy]",
        "V50 [%]", "V60 [%]", "V65 [%]", "V70 [%]", "V75 [%]", "V78 [%]"))

    log_structure_stats_header(structure_enum)

def print_structure_name(structure_enum):
    print("\nStructure\t{}".format(structure_enum.value))

def print_patient_stats_header():
    """ Print header for dose/volume statistics for structures """
    print("{:<10}\t{:<16}\t{:<15}"
          "\t{:<10}\t{:<10}\t{:<10}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
        "Patient ID", "ROI Name", "Volume [cm3]",
        "D_min [Gy]", "D_mean [Gy]", "D_max [Gy]", "D2cc [Gy]",
        "D2 [Gy]", "D50 [Gy]", "D90 [Gy]", "D98 [Gy]",
        "V50 [%]", "V60 [%]", "V65 [%]", "V70 [%]", "V75 [%]", "V80 [%]"))

    log_patient_stats_header()

def print_structure_stats(patient_id, roi_name, volume, dvh):
    """ Print dose/volume statistics for the structures and DVH passed """

    # ====== DEFINE WANTED DVH STATS ======
    dvh_stats = ('D2', 'D50', 'D90', 'D98',
                 'V50Gy', 'V60Gy', 'V65Gy', 'V70Gy', 'V75Gy', 'V78Gy',
                 )

    rel_dvh = dvh.relative_volume
    stats = [rel_dvh.statistic(s).value for s in dvh_stats]

    print("{:<10}\t{:<16}\t{:<15.1f}"
          "\t{:<10.2f}\t{:<10.2f}\t{:<10.2f}\t{:<8.2f}"
          "\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}"
          "\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}".format(
        patient_id, roi_name, volume,
        dvh.min, dvh.mean, dvh.max, dvh.D2cc.value, *stats))

    log_structure_stats(patient_id, roi_name, volume, dvh)


def log_structure_stats_header(structure_enum):
    log_structure_name(structure_enum)
    logger.info(
          "\n{:<10}\t{:<16}\t{:<15}"
          "\t{:<10}\t{:<10}\t{:<10}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
        "Patient ID", "ROI Name", "Volume [cm3]",
        "D_min [Gy]", "D_mean [Gy]", "D_max [Gy]",
        "D2cc [Gy]", "D2 [Gy]", "D50 [Gy]", "D60 [Gy]",
        "D70 [Gy]", "D90 [Gy]", "D95 [Gy]", "D98 [Gy]",
        "V50 [%]", "V60 [%]", "V65 [%]", "V70 [%]", "V75 [%]", "V80 [%]"))

def log_structure_name(structure_enum):
    logger.info("Structure\t{}".format(structure_enum.value))

def log_patient_stats_header():
    """ Print header for dose/volume statistics for structures """
    logger.info("{:<10}\t{:<16}\t{:<15}"
          "\t{:<10}\t{:<10}\t{:<10}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}"
          "\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
        "Patient ID", "ROI Name", "Volume [cm3]",
        "D_min [Gy]", "D_mean [Gy]", "D_max [Gy]",
        "D2cc [Gy]", "D2 [Gy]", "D50 [Gy]", "D60 [Gy]",
        "D70 [Gy]", "D90 [Gy]", "D95 [Gy]", "D98 [Gy]",
        "V50 [%]", "V60 [%]", "V65 [%]", "V70 [%]", "V75 [%]", "V80 [%]"))

def log_structure_stats(patient_id, roi_name, volume, dvh):
    """ Print dose/volume statistics for the structures and DVH passed """
    rel_dvh = dvh.relative_volume
    logger.debug("{:<10}\t{:<16}\t{:<15.1f}"
          "\t{:<10.2f}\t{:<10.2f}\t{:<10.2f}"
          "\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}"
            "\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}"
          "\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}"
            "\t{:<8.1f}\t{:<8.1f}\t{:<8.1f}".format(
        patient_id, roi_name, volume,
        dvh.min, dvh.mean, dvh.max,
        dvh.D2cc.value, rel_dvh.D2.value, rel_dvh.D50.value,
            rel_dvh.D60.value, rel_dvh.D70.value, rel_dvh.D90.value,
            rel_dvh.D95.value, rel_dvh.D98.value,
        rel_dvh.statistic('V50Gy').value, rel_dvh.statistic('V60Gy').value,
        rel_dvh.statistic('V65Gy').value, rel_dvh.statistic('V70Gy').value,
        rel_dvh.statistic('V75Gy').value, rel_dvh.statistic('V80Gy').value))


def print_lkb_params(td50, td50_std, n, n_std, m, m_std,
                     num_samples, confidence_level=0.95):
    """
    LL : float
        Log-likelihood of observation for the parameters
    """
    print("{:<8}\t{:} {:<4}\t{:<5}".format(
        "PARAMETER", "VALUE±STD", "", "+/- 1 STD"))
    print("* {:<6}\t{:.3f}±{:<8.3f}\t({:<.3f}-{:<.3f})".format(
        "TD50", td50, td50_std, td50 - td50_std, td50 + td50_std))
    print("* {:<6}\t{:.3f}±{:<8.3f}\t({:<.3f}-{:<.3f})".format(
        "n", n, n_std, n - n_std, n + n_std))
    print("* {:<6}\t{:.3f}±{:<8.3f}\t({:<.3f}-{:<.3f})".format(
        "m", m, m_std, m - m_std, m + m_std))

    from scipy import stats
    conf_int_td50 = stats.norm.interval(
        confidence_level, loc=td50, scale=td50_std / np.sqrt(num_samples))
    conf_int_n = stats.norm.interval(
        confidence_level, loc=n, scale=n_std / np.sqrt(num_samples))
    conf_int_m = stats.norm.interval(
        confidence_level, loc=m, scale=m_std / np.sqrt(num_samples))
    print("\n{:.0f} % CONFIDENCE INTERVALS".format(100*confidence_level))
    print("TD50: ({:.3f} - {:.3f})".format(conf_int_td50[0], conf_int_td50[1]))
    print("n: ({:.3f} - {:.3f})".format(conf_int_n[0], conf_int_n[1]))
    print("m: ({:.3f} - {:.3f})".format(conf_int_m[0], conf_int_m[1]))

    log_lkb_params(
        td50, td50_std, n, n_std, m, m_std, num_samples, confidence_level)

def log_lkb_params(td50, td50_std, n, n_std, m, m_std,
                   num_samples, confidence_level=0.95):
    logger.info("Optimized LKB model parameters")
    logger.info("{:<8}\t{:} {:<4}\t{:<5}".format(
        "PARAMETER", "VALUE±STD", "", "+/- 1 STD"))
    logger.info("* {:<6}\t{:.3f}±{:<8.3f}\t({:<.3f}-{:<.3f})".format(
        "TD50", td50, td50_std, td50 - td50_std, td50 + td50_std))
    logger.info("* {:<6}\t{:.3f}±{:<8.3f}\t({:<.3f}-{:<.3f})".format(
        "n", n, n_std, n - n_std, n + n_std))
    logger.info("* {:<6}\t{:.3f}±{:<8.3f}\t({:<.3f}-{:<.3f})".format(
        "m", m, m_std, m - m_std, m + m_std))

    from scipy import stats
    conf_int_td50 = stats.norm.interval(
        confidence_level, loc=td50, scale=td50_std / np.sqrt(num_samples))
    conf_int_n = stats.norm.interval(
        confidence_level, loc=n, scale=n_std / np.sqrt(num_samples))
    conf_int_m = stats.norm.interval(
        confidence_level, loc=m, scale=m_std / np.sqrt(num_samples))
    logger.info("TD50: ({:.3f} - {:.3f})".format(
        conf_int_td50[0], conf_int_td50[1]))
    logger.info("n: ({:.3f} - {:.3f})".format(conf_int_n[0], conf_int_n[1]))
    logger.info("m: ({:.3f} - {:.3f})".format(conf_int_m[0], conf_int_m[1]))


def timestamp(t=None):
    from time import strftime, localtime

    if t is None:
        t = localtime()
    return strftime('%Y-%m-%d-%H%M%S', t)


# =====================================
#   OTHER HELPER FUNCTIONS
# =====================================

def get_masked_dose_matrix(patient, structure_of_interest):
    """
    NOTE: This function is a work in progress...
    TODO: Test ConvexHull and remove helper plot when function is finished

    Calculates rectal wall dose matrix if RECTAL_WALL is structure of interest
        (SOI), otherwise searches Patient ROIs for the SOI and reads masked
        dose matrix.

    Returns
    -------
    masked_matrix : DoseMatrix
        DoseMatrix object containing the masked dose matrix for the given SOI
    roi_name : int
        ** The ROI name defined in patient, or
        ** "Rectal wall" for RECTAL_WALL
    roi_volume : float
        The volume of the ROI in mm3
    roi_id : float
        ** The ROI ID defined in patient, or
        ** "-" for RECTAL_WALL (has no ROI ID)
    """

    from dose.dosemask import get_mask_matrix, apply_roi_mask
    from utilities.exceptions import ROIError


    # If RECTAL WALL: Get DoseMatrix from rectum and rectal mucosa DoseMatrix
    if structure_of_interest == StructureEnum.RECTAL_WALL:

        # ===== Get ROIStructures for rectum/rectal mucosa =====
        try:
            rectum_id = find_roi_number_from_roi_name(
                patient, StructureEnum.RECTUM)
            mucosa_id = find_roi_number_from_roi_name(
                patient, StructureEnum.RECTAL_MUCOSA)
            rectum = patient.get_roi(rectum_id)
            mucosa = patient.get_roi(mucosa_id)
        except Exception as err:
            msg = "Could not find one of ROIs 'Rectum' or 'Rectal mucosa' for "\
                  "patient with ID {}. {}: {}".format(
                patient.get_id(), err.__class__.__name__, err)
            print(msg)
            logger.error(msg)
            raise ROIError(msg)

        # ===== Adjust ROIStructure with margins, etc =====
        '''rectum = rectum.to_convex_hull()
        mucosa = mucosa.to_convex_hull()
        rectum.expand_contours(margin=1., mode='absolute')   # expand rectum
        mucosa.expand_contours(margin=-1., mode='absolute')  # shrink mucosa'''


        # ===== Get rectum and rectal mucosa masks =====
        total_dose = patient.get_dose_matrix()
        rectum_mask = get_mask_matrix(total_dose, rectum)
        mucosa_mask = get_mask_matrix(total_dose, mucosa)

        # ===== Get rectal wall mask and calculate DoseMatrix =====
        rectal_wall_mask = np.logical_xor(rectum_mask, mucosa_mask)
        rectal_wall_dose = apply_roi_mask(total_dose, rectal_wall_mask)
        rectal_wall_dose.set_masking_data(roi_id="-", roi_name="Rectal wall")

        # Print volume calculations from different sources
        px = total_dose.pixel_spacing
        v_conv_hull_rectum = rectum.volume_convex_hull()
        v_conv_hull_mucosa = mucosa.volume_convex_hull()
        print("ORGAN\tVOXELS\tFROM MASK\tCONVHULL\tFROM PATIENT")
        print(# Rectum
              f"Rectum\t{np.sum(rectum_mask.astype(dtype=int)):<6}"
              f"\t{roi_volume_cm3(rectum_mask, px):<10}"
              f"\t{v_conv_hull_rectum:<10.3f}"
              f"\t{patient.get_roi_volume(rectum_id)}"
              # Rectal mucosa
              f"\nMucosa\t{np.sum(mucosa_mask.astype(dtype=int)):<6}"
              f"\t{roi_volume_cm3(mucosa_mask, px):<10}"
              f"\t{v_conv_hull_mucosa:<10.3f}"
              f"\t{patient.get_roi_volume(mucosa_id)}"
              # Rectal wall
              #f"\nR-wall\t{np.sum(rectal_wall_mask.astype(dtype=int)):<6}"
              #f"\t{roi_volume_cm3(rectal_wall_mask, px):<10}"
              #f"\t{'-':10}\t(no volume)"
              )

        # PLOT DOSE MATRIX HERE FOR VERIFICATION WHEN DEBUGGING
        plot_matrix = True
        if plot_matrix:
            import matplotlib.pyplot as plt

            SLICE_NO = 50

            DOSE_MIN = 0
            DOSE_MAX = 80
            FS_HEADER = 18
            dx = total_dose.x[1]-total_dose.x[0]
            dy = total_dose.y[1]-total_dose.y[0]

            xx, yy = np.meshgrid(total_dose.x, total_dose.y)
            r_dose = apply_roi_mask(total_dose, rectum_mask)
            rm_dose = apply_roi_mask(total_dose, mucosa_mask)
            rw_dose = rectal_wall_dose

            ct = patient.get_ct_slice(DoseGroup.TOTAL, SLICE_NO)
            ct_image = ct['image_sort']
            ct_xx, ct_yy = np.meshgrid(ct['x'], ct['y'])


            # ====== Create figure ======
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,5))
            #fig.suptitle(f"Patient ID: {patient.get_id()}", fontsize=FS_HEADER)
            (ax1, ax2, ax3) = axes

            for ax in axes:
                ct = ax.pcolormesh(ct_xx, ct_yy, ct_image,
                                   cmap='gray', shading='auto')

                ax.set_aspect(dy/dx)
                ax.set_axis_off()

                #ax.set_xlim(np.min(ct_xx), np.max(ct_xx))
                #ax.set_ylim(np.max(ct_yy), np.min(ct_yy))
                ax.set_xlim(-130, 130)
                ax.set_ylim(-75, -250)

            kwargs = {'cmap': 'plasma', 'shading': 'gouraud', 'alpha': 0.2,
                      'vmin': DOSE_MIN, 'vmax': DOSE_MAX}
            r = ax1.pcolormesh(xx, yy, r_dose.dose[SLICE_NO], **kwargs)
            rm = ax2.pcolormesh(xx, yy, rm_dose.dose[SLICE_NO], **kwargs)
            rw = ax3.pcolormesh(xx, yy, rw_dose.dose[SLICE_NO], **kwargs)

            try:
                r_contour = rectum.get_slice(total_dose.z[SLICE_NO])
                ax1.plot(r_contour.x, r_contour.y)
                rm_contour = mucosa.get_slice(total_dose.z[SLICE_NO])
                ax2.plot(rm_contour.x, rm_contour.y)
            except:
                pass

            ax1.set_title("Rectum", fontsize=FS_HEADER)
            ax2.set_title("Rectal mucosa", fontsize=FS_HEADER)
            ax3.set_title("Rectal wall", fontsize=FS_HEADER)

            from os import makedirs
            from os.path import isdir, join
            from constants import OUT_DATA_PATH
            plot_path = join(OUT_DATA_PATH, 'rectal-wall')
            if not isdir(plot_path):
                makedirs(plot_path)
            fn = f"rectal_wall-id_{patient.get_id()}-slice_{SLICE_NO}" \
                 f"" \
                 f"-{timestamp()}"
            plt.savefig(join(plot_path, fn))
            plt.show()


        roi_volume = \
            roi_volume_cm3(rectal_wall_dose.dose, total_dose.pixel_spacing)
        return rectal_wall_dose, "Rectal wall", roi_volume, "-"


    # If OTHER STRUCTURE: Try to find structure in already defined structures
    else:

        roi_id = find_roi_number_from_roi_name(patient, structure_of_interest)
        # Get masked dose matrix identified by ROI ID
        masked_matrix = patient.get_masked_dose_matrix(roi_id)
        roi_name = patient.get_roi_name(roi_id)
        roi_volume = patient.get_roi_volume(roi_id)

        return masked_matrix, roi_name, roi_volume, roi_id


def roi_volume_cm3(masked_matrix, px_spacing):
    """
    Takes in a masked matrix (could be the `dose` attribute of a masked
      DoseMatrix object or a ROI mask matrix) and returns the ROI volume
      defined by the mask.

    Parameters
    ----------
    masked_matrix : np.ma.maskedarray (usually 3D)
        Masked matrix (e.g. the `dose` attribute of a DoseMatrix object or a
        ROI mask matrix) from which to calculate volume.
    px_spacing : size-3 ndarray
        Pixel spacing in patient coordinates (in mm) in the (z, x, y) direction
    """

    if not isinstance(masked_matrix, np.ma.MaskedArray):
        masked_matrix = np.ma.masked_array(masked_matrix, mask=~masked_matrix)
    compressed_pts = masked_matrix.compressed()
    num_pts = np.size(compressed_pts)
    vol_mm3 = num_pts * px_spacing[0] * px_spacing[1] * px_spacing[2]
    vol_cm3 = vol_mm3 / 1000
    return vol_cm3


# =====================================
#   DVH AND VOXEL HELPER FUNCTIONS
# =====================================

def get_volumes_from_dvh_file(struct, patients=None):
    """
    Get all available DVHs (all patients that have data for the given struct).
        Return dictionary of DVHs indexed by patient ID.
    """

    from fileio.readwritedata import read_dvh_from_file

    if patients is None:
        patients = AVAILABLE_PATIENTS  # For all patients

    # ========= GET VOLUMES =========
    volumes = {}
    for patient_id in patients:

        if patient_id in NO_PATIENT_DATA:
            print("{:<10}\tData not available/Excluded".format(patient_id))
            continue

        try:
            # Read data from DVH file
            dvh_data = read_dvh_from_file(patient_id, struct)
            volumes[patient_id] = dvh_data['ROI volume']
        except FileNotFoundError as err:
            print("{:<10}\tNo DVH file found for ROI {}."
                  .format(patient_id, struct.name))
        except Exception as err:
            print("{:<10}\tCould not get DVH data for ROI {}. {}: {}"
                  "".format(patient_id, struct.name,
                            err.__class__.__name__, err))

    return volumes


def get_dvhs_from_file(struct, patients=None):
    """
    Get all available DVHs (all patients that have data for the given struct).
        Return dictionary of DVHs indexed by patient ID.
    """

    from fileio.readwritedata import dvh_from_file

    if patients is None:
        patients = AVAILABLE_PATIENTS  # For all patients

    # ========= GET DVHs =========
    dvhs = {}
    volumes = {}
    for patient_id in patients:

        if patient_id in NO_PATIENT_DATA:
            print("{:<10}\tData not available/Excluded".format(patient_id))
            continue

        try:
            # Read data from DVH file
            diff_dvh, roi_volume = dvh_from_file(patient_id, struct)
            cum_dvh = diff_dvh.cumulative
            rel_dvh = cum_dvh.relative_volume
            dvhs[patient_id] = rel_dvh
            volumes[patient_id] = roi_volume

        except FileNotFoundError as err:
            print("{:<10}\tNo DVH file found for ROI {}."
                  .format(patient_id, struct.name))
        except Exception as err:
            print("{:<10}\tCould not get DVH stats for ROI {}. {}: {}"
                  "".format(patient_id, struct.name,
                            err.__class__.__name__, err))

    return dvhs, volumes


def get_dvh_counts(dvh_array, pad_to=None):
    """
    Gets the number of counts from a list of DVHs. Pads counts arrays so that
        all arrays have the same length (allows use of numpy array operations)

    Parameters
    ----------
    dvh_array : array/list of dicompylercore.dvh.DVH objects
        Array of DVHs for several subjects
    pad_to : integer
        (optional) If defined, the counts arrays are padded to this length

    Returns
    -------
    counts_arr : array/list of ndarrays
        A list of arrays containing the number of counts in the DVHs.
    """

    # If pad_to = None, pad counts arrays to match longest array length
    if pad_to is None:
        pad_to = 0
        for dvh in dvh_array:
            num_counts = np.size(dvh.counts)
            if num_counts > pad_to:
                pad_to = num_counts

    # Read counts arrays, pad, and store in numpy array
    counts_arr = []
    for dvh in dvh_array:
        counts = dvh.counts
        counts = np.pad(counts, (0, pad_to - np.size(counts)))
        counts_arr.append(counts)
    return np.asarray(counts_arr)


def get_wanted_dvh_type(dvhs, volumes, dvh_type, dvh_volume_type, binsize):

    # Default DVH type is RELATIVE VOLUME, CUMULATIVE DVHs
    new_dvhs = {}
    for id, dvh in dvhs.items():
        # Set binsize
        if binsize != (dvh.bincenters[1]-dvh.bincenters[0]):
            dvh = change_dvh_binsize(dvh, new_binsize=binsize)
        # Differential/cumulative
        if dvh_type == 'differential':
            dvh = dvh.differential
        else:  # dvh_type == 'cumulative'
            dvh = dvh.cumulative
        # Absolute/relative volume
        if dvh_volume_type == 'absolute':
            dvh = dvh.absolute_volume(volumes[id])
        else:  # dvh_volume_type == 'relative':
            dvh = dvh.relative_volume
        new_dvhs[id] = dvh

    return new_dvhs


def change_dvh_binsize(old_dvh, new_binsize):
    """

    old_dvh : dicompylercore.dvh.DVH()
        The DVH for which you want a new binsize
    new_binsize : float
        Binsize (in Gy) of the new DVH
    :return:
    """
    from dicompylercore import dvh

    old_binsize = old_dvh.bins[1] - old_dvh.bins[0]
    if (old_binsize == new_binsize):
        return old_dvh

    # Create new counts and bins arrays from old DVH
    scale_factor = int(new_binsize / old_binsize)
    new_bins = old_dvh.bins[::scale_factor]

    num_counts = int(np.size(old_dvh.counts)/scale_factor)
    new_counts = np.zeros(shape=(num_counts))
    for i in range(num_counts):
        new_counts[i] = np.sum(old_dvh.counts[i*scale_factor : (i+1)*scale_factor])

    # Create new DVH from new counts and bins
    new_dvh = dvh.DVH(new_counts, new_bins).relative_volume
    new_dvh = new_dvh.cumulative
    return new_dvh


def read_voxel_data(patients, struct):
    """
    Reads voxel data files for all patients from outdata/voxeldata folder and
        pads the resulting dose-voxel array.

    Parameters
    ----------
    patients : list of ints
        The patients for which to read voxel data
    struct : StructureEnum
        Structure (ROI) to get data for

    Returns
    -------
    voxel_data : dictionary
        Dictionary of voxel-dose arrays, indexed by patient id
    """
    from fileio.readwritedata import read_voxel_data_from_file

    MAX_VOXELS = 20000
    voxel_data = {}

    for patient_id in patients:
        try:
            voxel_dataset = read_voxel_data_from_file(patient_id, struct)
            doses = voxel_dataset['dose']
            doses = np.pad(doses, (0, MAX_VOXELS - np.size(doses))) # pad with 0
            voxel_data[patient_id] = doses
        except FileNotFoundError as err:
            logger.info("{:<10}\tNo voxel data file found for ROI {}. "
                           "".format(patient_id, struct.name))

    return voxel_data


def sort_voxel_and_complication_data(voxel_data, complication_data, threshold):
    voxels = []
    outcome = []
    for id in AVAILABLE_PATIENTS:
        if ((id in voxel_data.keys()) and (id in complication_data.keys())):
            voxels.append(voxel_data[id])
            if complication_data[id] >= threshold:
                outcome.append(1)
            else:
                outcome.append(0)
    return voxels, outcome


# =====================================
#   PCA HELPER FUNCTIONS
# =====================================

def dataframe_by_arm(df):
    import pandas as pd
    arm1 = filter_study_arms(AVAILABLE_PATIENTS, wanted_arm=1)
    arm2 = filter_study_arms(AVAILABLE_PATIENTS, wanted_arm=2)
    arm1_df = pd.DataFrame(df, index=arm1)
    arm2_df = pd.DataFrame(df, index=arm2)
    return arm1_df, arm2_df


# ==============================================
#   REGISTERED IMAGES, DOSE ANALYSIS HELPERS
# ==============================================

def get_registered_dvhs(structure, patients=None, use_fixed_mask_id=None):
    """

    Get DVHs and ROI volume from registered dose distributions and mask. Use
    mask of each patient unless a mask ID is passed (e.g. the fixed patient
    ID). Get DVHs for all patients that have data for the given struct).

    Return dictionary of DVHs and volumes indexed by patient ID.

    """

    import os
    import SimpleITK as sitk
    import fileio.readwritedata as rwdata
    from constants import OUT_DATA_PATH


    if patients is None:
        patients = AVAILABLE_PATIENTS  # For all patients

    # ========= GET DATA =========
    if use_fixed_mask_id is not None:
        structure_mask_img = get_mask_from_file(structure, use_fixed_mask_id,
                                                is_fixed_patient=True)

    dvhs = {}
    volumes = {}
    for patient_id in patients:

        if patient_id in NO_PATIENT_DATA:
            print("{:<10}\tData not available/Excluded".format(patient_id))
            continue

        # File path to read/write registered DVHs
        if use_fixed_mask_id is None: # Using patient's own mask
            filepath = os.path.join(
                OUT_DATA_PATH, os.path.join('registered-DVH', structure.name))
        else: # Use the same mask for all patients
            filepath = os.path.join(
                OUT_DATA_PATH,
                os.path.join('registered-DVH-fixed-mask', structure.name))


        # Try to read DVH from existing file
        try:
            dvh, roi_volume = rwdata.dvh_from_file(
                patient_id, structure, file_path=filepath)
            dvhs[patient_id] = dvh.relative_volume
            volumes[patient_id] = roi_volume
            # Skips next try-except clause if file is found
            continue
        except FileNotFoundError as err:
            print(f"{patient_id:<10}"
                  f"\tNo existing DVH file found for ROI {structure.name}. "
                  f"Trying to generate DVH file.")

        # If reading from file fails, try to generate DVH from raw data
        try:
            # Read structure data from files
            if use_fixed_mask_id is None:
                structure_mask_img = get_mask_from_file(structure, patient_id)

            # Read dose data from file
            if use_fixed_mask_id == patient_id:
                dose_image = get_registered_dose_from_file(
                    patient_id, is_fixed_patient=True)
            else:
                dose_image = get_registered_dose_from_file(
                    patient_id, is_fixed_patient=False)

            # Mask dose array with the mask and get DVH
            dvh = diff_dvh_from_dose_and_mask_images(
                dose_image, structure_mask_img, roi_name=structure.value)

            # Calculate ROI volume
            roi_volume = roi_volume_cm3(
                sitk.GetArrayFromImage(structure_mask_img).astype(dtype=bool),
                px_spacing=structure_mask_img.GetSpacing())

            dvhs[patient_id] = dvh.relative_volume
            volumes[patient_id] = roi_volume

            # Write DVH and voxel data and generate file from masked matrix
            rwdata.write_dvh_to_file(
                patient_id, structure, roi_volume, dvh, file_path=filepath)

            print(f"{patient_id:<10}"
                  f"\tCalculated DVH and volume for {structure.name}.")

        except FileNotFoundError as err:
            print(f"{patient_id:<10}"
                  f"\tNo DVH file found for ROI {structure.name}.")
        except Exception as err:
            print(f"{patient_id:<10}"
                  f"\tCould not get DVH stats for ROI {structure.name}. "
                  f"{err.__class__.__name__}: {err}")

    return dvhs, volumes


def get_mask_from_file(structure, patient_id, is_fixed_patient=False):
    """

    Read a generated structure mask from file and return as SimpleITK.Image

    Parameters
    ----------
    structure : StructureEnum
        The structure the mask represents
    patient_id : int
        The patient identifier

    Returns
    -------
    mask_image : SimpleITK.Image
        The binary mask read from the file

    """

    import os
    import SimpleITK as sitk
    import constants

    path = constants.IMAGEREG_OUT_PATH
    if is_fixed_patient:    # Filename does not have '-transformed-nii' ending
        filename = constants.STRUCTURE_FILENAME.format(
            structure=structure.name, patient_id=patient_id)
    else:
        filename = constants.TRANSFORMED_STRUCTURE_FILENAME.format(
            structure=structure.name, patient_id=patient_id)
    mask_image = sitk.ReadImage(os.path.join(path, filename), sitk.sitkInt8)
    return mask_image


def get_registered_dose_from_file(patient_id, is_fixed_patient=False):
    """

    Read a dose distribution from file and return as SimpleITK.Image

    Parameters
    ----------
    patient_id : int
        The patient identifier

    Returns
    -------
    dose_image : SimpleITK.Image
        The dose distribution read from the file, as SimpleITK.Image

    """

    import os
    import SimpleITK as sitk
    import constants

    path = constants.IMAGEREG_OUT_PATH
    if is_fixed_patient:
        filename = constants.DOSE_FILENAME.format(patient_id=patient_id)
    else:
        filename = constants.TRANSFORMED_DOSE_FILENAME.format(patient_id=patient_id)
    dose_image = sitk.ReadImage(os.path.join(path, filename))
    return dose_image


def diff_dvh_from_dose_and_mask_images(dose_image, mask_image, roi_name):

    import SimpleITK as sitk
    from dvhcalculator import DVHCalculator

    # Convert to numpy array
    dose_array = sitk.GetArrayFromImage(dose_image)
    structure_mask = sitk.GetArrayFromImage(mask_image)
    structure_mask = structure_mask.astype(dtype=bool)

    # Mask dose array with the mask and get DVH
    masked_array = np.ma.masked_array(dose_array, mask=~structure_mask)
    dvh = DVHCalculator.differential_dvh_from_masked_array(
        masked_array, roi_name=roi_name)

    return dvh



# =====================================
#   OTHER HELPER FUNCTIONS
# =====================================

def get_study_arms():
    """
    Reads the 'studyarms.txt' file that specifies (patient ID, study arm) pairs
        and returns dictionary.

    Returns
    -------
    study_arms : dictionary
        Dictionary specifying study arm of each patient. Indexed by patient ID.
    """
    from os.path import join
    from constants import PROJECT_DATA_PATH

    f = open(join(PROJECT_DATA_PATH, 'studyarms.txt'))
    data = f.readlines()
    f.close()

    study_arms = {}
    for line in data[1:]:
        line = line.split()
        if len(line) == 1:  # No treatment arm given, excluded
            study_arms[int(line[0])] = None
        else:
            study_arms[int(line[0])] = int(line[1])

    return study_arms


def filter_study_arms(available_patients, wanted_arm):
    """
    Filter patients by study arm

    Parameters
    ----------
    available_patients: list of integers
        The available patient IDs to check study arm for
    wanted_arm : int/str ('both', 1 or 2 (integers))
        The desired study arm to investigate
    study_arms : dictionary
        Dictionary of study arms (1, 2 or None/empty), indexed by patient ID
    """

    study_arms = get_study_arms()
    if wanted_arm == 'both':    # Return list of all patient IDs
        return available_patients

    # Remove patient IDs that are not in wanted_arm
    for id in available_patients:
        if study_arms[id] != wanted_arm:
            available_patients = np.delete(
                available_patients, np.argwhere(available_patients == id))
    return available_patients


def get_treatment_centre():
    """
    Reads the 'centre.txt' file that specifies (patient ID, treatment centre)
    pairs and returns dictionary.

    Returns
    -------
    centre : dictionary
        Dictionary specifying centre patient was treated at. Indexed by ID.
    """

    from os.path import join
    from constants import PROJECT_DATA_PATH

    f = open(join(PROJECT_DATA_PATH, 'centre.txt'))
    data = f.readlines()
    f.close()

    centres = {}
    for line in data[1:]:
        line = line.split()
        id = int(line[0])
        if len(line) == 1:  # No treatment arm given, excluded
            centres[id] = None
        else:
            centre = int(line[1])
            if centre == 1:
                centres[id] = 'stolavs'
            elif centre == 2:
                centres[id] = 'aalesund'

    return centres


def filter_treatment_centre(available_patients, wanted_centre):
    """
    Filter patients by study arm

    Parameters
    ----------
    available_patients: list of integers
        The available patient IDs to check study arm for
    wanted_centre : 'stolavs', 'aalesund', or 'both'
        The desired centre
    study_arms : dictionary
        Dictionary of centre names, indexed by patient ID
    """

    centre = get_treatment_centre()
    if wanted_centre == 'both':    # Return list of all patient IDs
        return available_patients

    # Remove patient IDs that are not in wanted_arm
    for id in available_patients:
        if centre[id] != wanted_centre:
            available_patients = np.delete(
                available_patients, np.argwhere(available_patients == id))
    return available_patients


def count_patients_exceeding_V50Gy_V60Gy(voxel_data, struct):
    """
    Count number of patients exceeding dose-volume constraints
    """

    if not (struct == StructureEnum.RECTUM or
            struct == StructureEnum.RECTAL_MUCOSA):
        return

    num_above_50 = 0
    num_above_60 = 0
    for k, vd in voxel_data.items():
        vd = vd[np.where(vd > 0.001)]
        if vd[np.size(vd) // 2] > 50: num_above_50 += 1
        if vd[np.size(vd) // 2] > 60: num_above_60 += 1
    print("Number violating V50Gy <= 50% for {}: {}/{}".format(
        struct, num_above_50, len(voxel_data)))
    print("Number violating V60Gy <= 50% for {}: {}/{}".format(
        struct, num_above_60, len(voxel_data)))


# =====================================
#   TEST FUNCTIONS
# =====================================

def test_list_files():
    from constants import DATA_PATH
    dir = DATA_PATH
    print(list_dicom_files_in_dir(dir))
    #print(list_dicom_files_in_dir_full(dir))

