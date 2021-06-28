"""
Created on 10/11/2020
@author: ingridtveten

TODO: Description...
"""

import numpy as np
from os import makedirs
from os.path import join, isdir
from dicompylercore.dvh import DVH

from patient import Patient
from utilities.util import StructureEnum
from constants import DVH_FILE_PATH, DVH_FILE_NAME, \
    DELIMITER, DVH_NUM_DECIMALS, BOOST_DOSE, DOSE_VOXEL_PATH, \
    DOSE_VOXEL_FILE_NAME


# ===========================
#   WRITE/READ DVH DATA
# ===========================

def write_dvh_to_file(patient_id, roi_structure, roi_volume, dvh,
                      file_path=None):
    """
    Writes DVH data & metadata (e.g. ROI info, DVH type, DVH units) to file

    Parameters:
    -----------
    patient_id : int/str
        The identifier for the patient
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    dvh : DVH (dicompylercore.dvh.DVH)
        The DVH instance containing data that should be written to file
    """

    if not (isinstance(roi_structure, StructureEnum) and isinstance(dvh, DVH)):
        pass

    # =========== OPEN FILE PATH ============
    if file_path is None:
        file_path = get_dvh_file_path(roi_structure)

    if not isdir(file_path):
        makedirs(file_path)
    filename = get_dvh_file_name(patient_id, roi_structure)
    file = join(file_path, filename)
    f = open(file, mode='w')


    # ======== WRITE FILE HEADER (DVH metadata) =========
    header =  'Patient ID'  + DELIMITER + str(patient_id) + '\n'
    header += 'ROI volume'  + DELIMITER + str(roi_volume) + '\n'
    header += 'ROI name'    + DELIMITER + roi_structure.name + '\n'
    header += 'DVH type'    + DELIMITER + dvh.dvh_type + '\n'
    header += 'Dose units'  + DELIMITER + dvh.dose_units + '\n'
    header += 'Volume units'+ DELIMITER + dvh.volume_units + '\n'
    header += 'Bin size'    + DELIMITER + str(dvh.bins[1] - dvh.bins[0]) + '\n'
    f.write(header)


    # =========== WRITE DVH DATA ============
    bins = dvh.bins
    counts = dvh.counts
    bincenters = dvh.bincenters

    f.write('counts{}bincenter{}binmin{}binmax\n'.format(
        DELIMITER, DELIMITER, DELIMITER, DELIMITER))
    for i in range(np.size(counts)):
        entry =  str(np.round(counts[i], DVH_NUM_DECIMALS)) + DELIMITER
        entry += str(np.round(bincenters[i], DVH_NUM_DECIMALS)) + DELIMITER
        entry += str(np.round(bins[i], DVH_NUM_DECIMALS)) + DELIMITER
        entry += str(np.round(bins[i+1], DVH_NUM_DECIMALS)) + '\n'
        f.write(entry)

    f.close()


def read_dvh_from_file(patient_id, roi_structure, file_path=None):
    """
    Reads DVH data file and returns metadata and voxel data as a dictionary.

    Parameters:
    -----------
    patient_id : int/str
        The identifier for the patient
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    """

    if not isinstance(roi_structure, StructureEnum):
        pass

    # =========== OPEN FILE PATH AND READ DATA ============
    if file_path is None:
        file_path = get_dvh_file_path(roi_structure)
    filename = get_dvh_file_name(patient_id, roi_structure)
    file = join(file_path, filename)
    f = open(file, mode='r')
    data = f.readlines()
    f.close()
    num_lines = len(data)


    # ======== READ FILE HEADER (DVH metadata) =========
    dvh_data = {}
    i = 0
    while i < 7:
        line = data[i].strip('\n')
        line = line.split(sep=DELIMITER)
        if (line[0] == 'ROI volume' or line[0] == 'Bin size'):
            dvh_data[line[0]] = float(line[1])
        else:
            dvh_data[line[0]] = line[1]
        i += 1
    # i should be 7


    # =========== INITIALIZE EMPTY DVH DATA CONTAINERS ============
    line = data[i].strip('\n')
    line = line.split(sep=DELIMITER)
    data_headers = []
    # counts    bincenter   binmin  binmax
    for j in range(len(line)):
        header = line[j]
        data_headers.append(header)
        dvh_data[header] = []
    i += 1


    # =========== READ DVH DATA ============
    while i < num_lines:
        line = data[i]
        line = line.strip('\n')
        line = line.split(sep=DELIMITER)
        for j in range(len(data_headers)):
            header = data_headers[j]
            data_entry = float(line[j])
            dvh_data[header].append(data_entry)
        i += 1


    # Only bin lower and upper limits are stored, make bin array
    dvh_data['bins'] =\
        np.array(dvh_data['binmin'] + dvh_data['binmax'][-1::])
    dvh_data['counts'] = np.array(dvh_data['counts'])

    return dvh_data


def make_dvh_from_file_data(dvh_data):
    patient_id = dvh_data['Patient ID']
    roi_volume = float(dvh_data['ROI volume'])
    roi_name = dvh_data['ROI name']
    dvh_type = dvh_data['DVH type']
    dose_units = dvh_data['Dose units']
    volume_units = dvh_data['Volume units']
    binsize = float(dvh_data['Bin size'])

    bins = dvh_data['bins']
    counts = dvh_data['counts']

    the_dvh = DVH(counts=counts,
                  bins=bins,
                  dvh_type=dvh_type,
                  name=roi_name,
                  dose_units=dose_units,
                  volume_units=volume_units,
                  rx_dose=BOOST_DOSE
                  )
    return the_dvh, roi_volume


def dvh_from_file(patient_id, roi_structure, file_path=None):
    dvh_data = read_dvh_from_file(patient_id, roi_structure,
                                  file_path=file_path)
    the_dvh, roi_volume = make_dvh_from_file_data(dvh_data)
    return the_dvh, roi_volume


# =============================
#   WRITE/READ DOSE VOXEL DATA
# =============================

def write_voxel_data_to_file(patient, roi_structure, masked_matrix,
                             roi_volume, roi_id=None):
    """
    Writes voxel data & metadata (e.g. number of voxels and dimensions) to file

    Parameters:
    -----------
    patient_id : int/str
        The identifier for the patient
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    roi_id : StructureEnum
        The ROI ID to write to file
    """

    if not (isinstance(roi_structure, StructureEnum)
            and isinstance(patient, Patient)):
        pass
    patient_id = patient.get_id()
    if roi_id is None:
        roi_id = "-"

    # =========== OPEN FILE PATH ============
    file_path = get_dose_voxel_file_path(roi_structure)
    if not isdir(file_path):
        makedirs(file_path)
    filename = get_dose_voxel_file_name(patient_id, roi_structure)
    file = join(file_path, filename)
    f = open(file, mode='w')


    # ======== GET PATIENT DATA =========
    px_spacing = patient.get_dose_matrix().pixel_spacing
    dose_points = masked_matrix.dose.compressed()
    sorted_dose = np.sort(dose_points)
    num_voxels = np.size(dose_points)


    # ======== WRITE FILE HEADER (Dose/volume metadata) =========
    # Num elements = 8
    header =  'Patient ID'  + DELIMITER + str(patient_id) + '\n'
    header += 'ROI ID'      + DELIMITER + str(roi_id) + '\n'
    header += 'ROI volume'  + DELIMITER + str(roi_volume) + '\n'
    header += 'ROI name'    + DELIMITER + roi_structure.name + '\n'
    header += 'Pixel spacing' + DELIMITER + str(px_spacing[0]) \
              + DELIMITER + str(px_spacing[1]) \
              + DELIMITER + str(px_spacing[2]) + '\n'
    header += 'Spatial units'+ DELIMITER + 'mm' + '\n'
    header += 'Dose units'  + DELIMITER + patient.get_dose_matrix().units + '\n'
    header += 'Number of voxels' + DELIMITER + str(num_voxels) + '\n'
    f.write(header)


    # =========== WRITE VOXEL DATA ============

    f.write('point{}dose\n'.format(DELIMITER))
    for i in range(num_voxels):
        entry =  str(i) + DELIMITER + \
                 str(np.round(sorted_dose[i], DVH_NUM_DECIMALS)) + '\n'
        f.write(entry)

    f.close()


def read_voxel_data_from_file(patient_id, roi_structure):
    """
    Reads voxel data file and returns metadata and voxel data as a dictionary.

    Parameters:
    -----------
    patient_id : int/str
        The identifier for the patient
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    roi_id : StructureEnum
        The ROI ID to write to file
    """

    if not isinstance(roi_structure, StructureEnum):
        pass

    # =========== OPEN FILE PATH ============
    file_path = get_dose_voxel_file_path(roi_structure)
    filename = get_dose_voxel_file_name(patient_id, roi_structure)
    file = join(file_path, filename)
    f = open(file, mode='r')
    data = f.readlines()
    f.close()
    num_lines = len(data)


    # ======== READ FILE HEADER (Dose/volume metadata) =========
    # Num elements = 8
    voxel_data = {}
    i = 0
    while i < 8:
        line = data[i].strip('\n')
        line = line.split(sep=DELIMITER)
        if (line[0] == 'ROI volume'):
            voxel_data[line[0]] = float(line[1])
        elif (line[0] == 'Number of voxels'):
            voxel_data[line[0]] = int(line[1])
        elif (line[0] == 'Pixel spacing'):
            voxel_data[line[0]] = [float(elem) for elem in line[1:]]
        else:
            voxel_data[line[0]] = line[1]
        i += 1
    # i should be 8


    # =========== INITIALIZE EMPTY DATA CONTAINERS ============
    line = data[i].strip('\n')
    line = line.split(sep=DELIMITER)
    data_headers = []
    # point     dose
    for j in range(len(line)):
        header = line[j]
        data_headers.append(header)
        voxel_data[header] = []
    i += 1


    # =========== READ VOXEL DATA ============
    while i < num_lines:
        line = data[i]
        line = line.strip('\n')
        line = line.split(sep=DELIMITER)
        for j in range(len(data_headers)):
            header = data_headers[j]
            data_entry = float(line[j])
            voxel_data[header].append(data_entry)
        i += 1

    # Make voxel data into numpy array
    voxel_data['point'] = np.array(voxel_data['point'])
    voxel_data['dose'] = np.array(voxel_data['dose'])

    return voxel_data



# ===========================
#   FILE NAMES
# ===========================

def get_dvh_file_path(roi_structure):
    """
    Parameters:
    -----------
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    """
    file_path = join(DVH_FILE_PATH, roi_structure.name)
    return file_path

def get_dvh_file_name(patient_id, roi_structure):
    """
    Parameters:
    -----------
    patient_id : int/str
        The identifier for the patient
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    """
    filename = DVH_FILE_NAME.format(patientid=patient_id,
                                    organ=roi_structure.name)
    return filename


def get_dose_voxel_file_path(roi_structure):
    """
    Parameters:
    -----------
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    """
    file_path = join(DOSE_VOXEL_PATH, roi_structure.name)
    return file_path

def get_dose_voxel_file_name(patient_id, roi_structure):
    """
    Parameters:
    -----------
    patient_id : int/str
        The identifier for the patient
    roi_structure : StructureEnum
        The type of ROI structure that the DVH represents
    """
    filename = DOSE_VOXEL_FILE_NAME.format(patientid=patient_id,
                                           organ=roi_structure.name)
    return filename




