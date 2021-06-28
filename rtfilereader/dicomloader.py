"""
Created on 28/09/2020
@author: ingridtveten

The DICOMLoader class loads data (with the specified patientID) from the folder
    specified by the DATA_PATH constant and sorts data by DICOMModality
"""

from os.path import join
import pydicom

from utilities.util import list_dicom_files_in_dir, DICOMModality
from constants import DATA_PATH


import logging
logger = logging.getLogger(__name__)


class DICOMLoader(object):
    """
    Reads DICOM files from directory and sorts the data by DICOM modality

    Attributes
    ----------
    patient_id: str
        The patient ID
    dir_path: string, optional
        Path to directory that files are loaded from (defaults to './data/')
    """

    patient_id = None
    dir_path = None

    def __init__(self, patient_id, dir_path=None):
        """ Constructor for DICOMLoader """
        logger.debug("Initialized new DICOMLoader object")
        self.patient_id = patient_id
        self.set_path(get_patient_folder(patient_id))

    def set_path(self, new_path): self.dir_path = new_path

    def get_path(self): return self.dir_path

    def load_patient_data(self):
        """
        Loads all DICOM files in the specified self.dir_path folder to a list
        """
        dcm_list = []

        dcm_path_list = list_dicom_files_in_dir(self.dir_path)
        for file in dcm_path_list:
            dataset = pydicom.dcmread(file)
            if (str(dataset.PatientName).lower() != 'ric'):
                print("Patient ID\t{}\nDICOM UID\t{}\nPatient name is not RIC")
            id = dataset.PatientID
            if (id == self.patient_id):
                dcm_list.append(dataset)

        return dcm_list

    def sort_data_by_modality(self, dcm_list):
        """
        Takes in list of pydicom.Dataset objects for the patient and returns
            dict where Dataset objects are listed and indexed by DICOMModality
        """
        sorted_dict = {}

        for dataset in dcm_list:
            modality = DICOMModality(dataset.Modality)
            if modality in sorted_dict: # If entry exists, append to list
                sorted_dict[modality].append(dataset)
            else: # Else create new entry (list)
                sorted_dict[modality] = [dataset]

        return sorted_dict


def get_patient_folder(patient_id):
    return join(DATA_PATH, str(patient_id))
