"""
Created on 30/09/2020
@author: ingridtveten
"""

import numpy as np
from os.path import join
import logging
from time import time

from rtfilereader.dicomloader import DICOMLoader, get_patient_folder
from rtfilereader.datasetreader import CTReader, RTStructureReader, RTDoseReader
from dose.dosematrix import DoseMatrix
from dose.dosemask import apply_roi_mask, get_mask_matrix
from utilities.exceptions import ROIMaskError, DoseMaskingError
from utilities.util import DoseGroup, DICOMModality


logger = logging.getLogger(__name__)


class Patient(object):
    """
    Class containing data belonging to a patient

    Attributes
    ----------
    patient_id: int/str
        Patient identifier
    """
    patient_id = None
    data = None
    dcm_loader = None


    def __init__(self, patient_id):#, patient_path=None):
        """ Constructor for Patient """
        t0 = time()
        # Initialize
        self.patient_id = str(patient_id)
        self.initialize_file_loading()#patient_path)
        self.initialize_data_containers()

        t1 = time()
        logger.debug("Initializing patient took {:.4f} s".format(t1-t0))

        # Read patient data
        self.read_data_series()
        self.copy_data_to_total_dose_series()
        # Get total dose data and masked dose matrices
        self.calculate_total_dose_matrix()

        t2 = time()
        logger.debug("Reading patient data and getting dose matrix took "
                     "{:.4f} s".format(t2-t1))

        #self.calculate_roi_masks()
        #self.mask_dose_matrices()

        t3 = time()
        logger.debug("Masking dose matrices took {:.4f} s".format(t3-t2))

        # Log/print for debugging purposes
        logger.debug("Initialized new patient: {}".format(self))
        self.log_data_element_stats()
        self.log_dose_matrix_stats()


    def __str__(self):
        """ String representation for Patient """
        txt = "Patient( ID: {}, Name: {}, Unit: {} )".format(
            self.get_id(),
            self.data[DoseGroup.BASE][DICOMModality.CT][0].PatientName,
            self.data[DoseGroup.BASE][DICOMModality.CT][0].InstitutionName
        )
        return txt


    """============================
        INITIALIZATION FUNCTIONS
    ============================"""
    def initialize_file_loading(self):
        """ Initializes the DICOM file loader and file path """
        self.patient_path = get_patient_folder(self.get_id())
        self.dcm_loader = DICOMLoader(self.get_id())


    def initialize_data_containers(self):
        self.data = {}
        self.ct_data = {}
        self.sorted_ct_data = {}
        self.rt_structures = {}

        self.total_dose = {}
        self.roi_masks = {}
        #self.masked_dose_matrix = {}


    def copy_data_to_total_dose_series(self):
        """ Make complete dataset (e.g. CT/ROI for plots) for total dose """
        # Should the data (incl. DOSE) be copied?
        #self.data[DoseGroup.TOTAL] = self.data[DoseGroup.BASE]
        # CT data
        self.ct_data[DoseGroup.TOTAL] = self.ct_data[DoseGroup.BASE]
        self.sorted_ct_data[DoseGroup.TOTAL] =\
                self.sorted_ct_data[DoseGroup.BASE]
        # ROI/structure data
        self.rt_structures[DoseGroup.TOTAL] =\
                self.rt_structures[DoseGroup.BASE]


    """============================
        READ/INITIALIZE DATA
    ============================"""
    def read_data_series(self):
        """ Load patient data by dose group and modality """
        for g in [DoseGroup.BASE, DoseGroup.BOOST]:

            # Get list of DICOM files and sorted data
            doseplan_path = join(self.patient_path, "DPL")
            self.dcm_loader.set_path(join(doseplan_path, g.value))
            dcm_list = self.dcm_loader.load_patient_data()
            self.data[g] = self.dcm_loader.sort_data_by_modality(dcm_list)

            # Get CT, ROI and dose data
            self.read_ct_data(g)
            self.read_rt_structures(g)
            try:
                self.read_dose_group_matrix(g)
            except AssertionError as err:
                logger.error("Error while reading dose files for patient {}. "
                             "Error message: {}".format(self.get_id(), err))
                exit()


    def read_ct_data(self, dose_group):
        """ Get CT images and metadata (format in Kajsa's script) """
        self.ct_data[dose_group] =\
            CTReader.get_ct_data(self.data[dose_group])
        self.sorted_ct_data[dose_group] =\
            CTReader.get_sorted_ct_data(self.data[dose_group])


    def read_rt_structures(self, dose_group):
        """ Get ROI structures from raw data """
        self.rt_structures[dose_group] = \
            RTStructureReader.get_rt_structures(self.data[dose_group])


    def read_dose_group_matrix(self, dose_group):
        """ Reads RTDOSE data and returns total dose as DoseGrid """
        dose_group_data = self.data[dose_group]
        # Check description for name of the dose group
        for dose in dose_group_data[DICOMModality.RTDOSE]:
            doseplan_name = dose.SeriesDescription
            if dose_group == DoseGroup.BASE:
                assert ("0" in doseplan_name and "70" in doseplan_name), \
                    ("Dose series description {} does not match group '0-70' "
                     "for UID {}".format(doseplan_name, dose.SOPInstanceUID))
            elif dose_group == DoseGroup.BOOST:
                assert ("70" in doseplan_name and "78" in doseplan_name), \
                    ("Dose series description {} does not match group '70-78' "
                     "for UID {}".format(doseplan_name, dose.SOPInstanceUID))

        total_matrix = RTDoseReader.get_dose_matrix(dose_group_data)
        self.total_dose[dose_group] = total_matrix


    def calculate_total_dose_matrix(self):
        """ Get BASE and BOOST dose matrices and sum these to make TOTAL """
        # Get base and boost matrices
        base = self.get_dose_matrix(dose_group=DoseGroup.BASE)
        boost = self.get_dose_matrix(dose_group=DoseGroup.BOOST)

        # Copy attributes from the BASE dose matrix and sum total dose
        self.total_dose[DoseGroup.TOTAL] = DoseMatrix.copy(base)
        self.total_dose[DoseGroup.TOTAL].dose = base.dose + boost.dose

        # Print/log stats for debugging purposes
        #self.print_dose_matrix_stats()
        self.log_dose_matrix_stats()


    def calculate_roi_masks(self):
        """ For each ROI, get the ROI mask for the dose matrix """
        for roi_id in self.get_roi_number_list():
            self.calculate_roi_mask(roi_id)


    def calculate_roi_mask(self, roi_id):
        """ For each ROI, get the ROI mask for the dose matrix """
        if roi_id not in self.roi_masks.keys():
            try:
                self.roi_masks[roi_id] = get_mask_matrix(
                    self.get_dose_matrix(), self.get_roi(roi_id))
            except IndexError as err:
                logger.error("IndexError calculating ROI mask. {}".format(err))
                raise ROIMaskError(err)
            except Exception as err:
                logger.error("Exception calculating ROI mask. {}".format(err))
                raise ROIMaskError(err)


    """============================
        GET FUNCTIONS
    ============================"""
    def get_id(self):
        return self.patient_id


    def get_ct_data(self, dose_group=DoseGroup.TOTAL, sorted=True):
        """ Get CT images """
        if sorted:
            return self.sorted_ct_data[dose_group]
        return self.ct_data[dose_group]


    def get_ct_slice(self, dose_group, slice_index):
        """ct_uid = self.sorted_ct_data[dose_group][slice_index]['uid']
        return self.ct_data[dose_group][ct_uid]"""
        return self.sorted_ct_data[dose_group][slice_index]


    def get_rt_structures(self, dose_group=None):
        """ Get ROI structures """
        if dose_group is not None:
            return self.rt_structures[dose_group]
        return self.rt_structures[DoseGroup.TOTAL]


    def get_rt_structures_dataset(self, dose_group):
        """ Get ROI structures """
        return self.data[dose_group][DICOMModality.RTSTRUCT][0]


    def get_roi_numbers(self):
        """ Assumes same keys in BASE and BOOST dose groups, returns BASE """
        return self.get_rt_structures().keys()


    def get_roi(self, roi_id, dose_group=None):
        """ Get ROIStructure by ROI ID """
        if dose_group is None:
            dose_group = DoseGroup.TOTAL
        return self.get_rt_structures()[roi_id]


    def get_roi_number_list(self):
        """ Assumes same keys in BASE and BOOST dose groups, returns BASE """
        roi_ids = []
        for roi in self.get_roi_numbers():
            roi_ids.append(roi)
        return roi_ids


    def get_roi_name(self, roi_number):
        if roi_number in self.get_roi_numbers():
            return self.get_rt_structures()[roi_number].name
        return None


    def get_total_dose_grid(self):
        """ Returns total dose as DoseGrid """
        return self.total_dose

    # Only used in in code from dicompyler.dvhcalc
    def get_dose_group(self, dose_group):
        """ Sums total dose using DoseGrid and returns as pydicom.Dataset """
        total_dose = self.get_total_dose_grid()[dose_group]
        return total_dose.ds


    def get_dose_matrix(self, dose_group=None):
        """
        Calculates DoseMatrix for given DoseGroup (or TOTAL if not specified)

        Returns
        -------
        DoseMatrix
            DoseMatrix corresponding to the dose_group (TOTAL if not specified)
        """
        if dose_group is None:
            dose_group = DoseGroup.TOTAL
        return self.total_dose[dose_group]
        #return self.dose_matrix[dose_group]


    def get_xyz_dose_matrix(self, dose_group=None):
        """ Calculates DoseMatrix for given DoseGroup with x-major axis """
        dose_matrix = self.get_dose_matrix(dose_group)
        # Swap axes to (x,y,z) from patient indexing (z,y,x)
        dose_matrix.dose = np.swapaxes(dose_matrix.dose, 0, 2)
        return dose_matrix


    def get_masked_dose_matrix(self, roi_number):
        """
        Returns the dose matrix masked by the given ROI

        Parameters:
        -----------
        roi_number: str
            The ROI id that specifies the masked DoseMatrix
        """

        try:
            roi_mask = self.get_roi_mask(roi_number)
        except ROIMaskError as err:
            logger.error("Error getting ROI mask. Returning None.".format(
                roi_number, self.get_id()))
            return None

        try:
            if roi_mask is not None:
                total_dose_matrix = self.get_dose_matrix()
                masked_matrix = apply_roi_mask(total_dose_matrix, roi_mask)
                masked_matrix.set_masking_data(roi_number,
                                               self.get_roi_name(roi_number))
                return masked_matrix
        except Exception as err:
            msg = "Error masking dose matrix for {} - {} for patient with ID " \
                  "{}. {}".format(roi_number, self.get_roi_name(roi_number),
                                  self.get_id(), err)
            logger.error(msg)
            raise DoseMaskingError(msg)


    def get_roi_mask(self, roi_number):
        """
        Returns the dose mask for the given ROI

        Parameters:
        -----------
        roi_number: str
            The ROI id that specifies the mask
        """
        if not roi_number in self.roi_masks.keys():
            self.calculate_roi_mask(roi_number)
        return self.roi_masks[roi_number]


    def get_roi_volume(self, roi_number):
        masked_array = np.ma.array(data=self.get_roi_mask(roi_number))
        compressed_pts = masked_array.compressed()
        num_pts = np.sum(compressed_pts.astype(dtype=int))
        px_spacing = self.get_dose_matrix().pixel_spacing
        vol_mm3 = num_pts * px_spacing[0] * px_spacing[1] * px_spacing[2]
        vol_cm3 = vol_mm3 / 1000
        return vol_cm3


    """============================
        OTHER FUNCTIONS
    ============================"""
    def save_total_dose_to_dcm(self, filepath):
        self.get_total_dose_grid()
        self.total_dose.save_dcm(file_path=filepath)


    def print_data_element_stats(self):
        num_dcms = 0
        print("Patient.data contains the keys:")
        for key in self.data.keys():
            num_elem = len(self.data[key])
            print("\t{:<24}{} elements".format(key, num_elem))
            num_dcms += num_elem
        print("\t{:>24}{}".format("Total = ", num_dcms))


    def log_data_element_stats(self):
        num_dcms = 0
        logger.debug("\nPatient.data contains the keys:")
        for key in self.data.keys():
            num_elem = len(self.data[key])
            logger.debug("\t{:<24}{} elements".format(key, num_elem))
            num_dcms += num_elem
        logger.debug("\t{:>24}{}".format("Total = ", num_dcms))


    def print_dose_matrix_stats(self):
        """ Debugging print statements """
        base = self.get_dose_matrix(DoseGroup.BASE)
        boost = self.get_dose_matrix(DoseGroup.BOOST)
        total = self.get_dose_matrix(DoseGroup.TOTAL)
        names = ("BASE", "BOOST", "TOTAL")

        print("{:<8}\t{:<6}\t{:<8}\tID: {}".format(
            "Dose [Gy]", "Min", "Max", self.get_id()))
        for i, group in enumerate([base, boost, total]):
            print("{:<8}\t{:<6}\t{:<8.5f}".format(
                names[i], group.min_dose, group.max_dose))


    def log_dose_matrix_stats(self):
        """ Debugging print statements """
        base = self.get_dose_matrix(DoseGroup.BASE)
        boost = self.get_dose_matrix(DoseGroup.BOOST)
        total = self.get_dose_matrix(DoseGroup.TOTAL)
        names = ("BASE", "BOOST", "TOTAL")

        logger.debug("{:<8}\t{:<6}\t{:<8}\tID: {}".format(
            "Dose [Gy]", "Min", "Max", self.get_id()))
        for i, group in enumerate([base, boost, total]):
            logger.debug("{:<8}\t{:<6}\t{:<8.5f}".format(
                names[i], group.min_dose, group.max_dose))


    def print_roi_id_and_names(self):
        print("Patient ID: {}".format(self.get_id()))
        for i in self.get_roi_numbers():
            print("{:<3}\t{}".format(i, self.get_roi_name(i)))


# ============================
#   PATIENT TEST FUNCTIONS
# ============================

def test_read_data():
    patient_id = 10
    patient = Patient(patient_id)
    patient.print_data_element_stats()


def test_patient():
    patient_id = 10
    patient = Patient(patient_id)
    for k, v in patient.data.items():
        print("{}\t{}".format(k, v))


if __name__ == "__main__":
    t1 = time()

    test_read_data()
    test_patient()

    t2 = time()
    logger.info("Process took {} s".format(t2-t1))