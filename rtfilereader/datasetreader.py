"""
Created on 05/10/2020
@author: ingridtveten

TODO: Docstring
"""

import numpy as np
import logging

from dose.dosegrid import DoseGrid
from dose.dosematrix import DoseMatrix
from utilities.util import DICOMModality, get_scaled_image


logger = logging.getLogger(__name__)


class DatasetReader(object):
    """ Parent class for patient analyzers """
    pass


class CTReader(DatasetReader):
    """ Class for analyzing and organizing CT data for a patient """

    @classmethod
    def get_ct_data(cls, patient_dataset):
        """ Load CT images and metadata (format in Kajsa's script) """
        data = cls.get_data(patient_dataset)
        ct_data = {}

        for dataset in data:
            ct_data[dataset.SOPInstanceUID] = {
                'ImagePositionPatient': # position of upper left voxel in patient coord
                    dataset.ImagePositionPatient,
                'PixelSpacing': # pixelsize in mm
                    dataset.PixelSpacing,
                'SliceLocation': dataset.SliceLocation,
                'image': get_scaled_image(dataset), # the actual image
                'dataset': dataset
            }
        return ct_data

    @classmethod
    def get_data(cls, patient_dataset):
        return patient_dataset[DICOMModality.CT]

    @classmethod
    def get_sorted_ct_data(cls, patient_dataset):
        """
        Sorts CT data by 'SliceLocation' and returns a list of dicts with
            the image, slice location and uid (to connect with ROIs)
        """
        ct_data = cls.get_ct_data(patient_dataset)
        sorted_ct_data = []
        for uid in ct_data: # loop over slices and store info
            ct = ct_data[uid]
            ct_ipp = ct['ImagePositionPatient']
            ct_px = ct['PixelSpacing']

            sorted_ct_data.append({
                'image_sort': ct['image'],
                'x': np.asarray([ (ct_ipp[0] + i * ct_px[0])
                                for i in range(ct['dataset'].Columns)]),
                'y': np.asarray([ (ct_ipp[1] + i * ct_px[1])
                                for i in range(ct['dataset'].Rows)]),
                'location_sort': ct['SliceLocation'],
                'uid': uid})

        def my_sort(x): return x['location_sort']
        sorted_ct_data.sort(key=my_sort)

        return sorted_ct_data


class RTStructureReader(DatasetReader):
    """ Class for getting RT structure set info from RTSTRUCT files """

    @classmethod
    def get_rt_structures(cls, patient_dataset):
        from rtfilereader.roistructure import ROIStructure

        # Initialize empty dictionary for RT structures
        structure_set = cls.get_structure_sets(patient_dataset)
        roi_info = cls.get_roi_info(structure_set)
        rt_structures = {}

        # Iterate through the contours in the structure set
        for contour_id in roi_info.keys():

            contour_data = cls.get_contour_sequence(structure_set, contour_id)
            assert (contour_id == contour_data.ReferencedROINumber)

            roi = ROIStructure(
                name=cls.get_contour_name(structure_set, contour_id),
                roi_number=contour_id,
                color=cls.get_contour_color(structure_set, contour_id),
                contour_data=contour_data
            )
            rt_structures[contour_id] = roi

        return rt_structures

    @classmethod
    def get_structure_sets(cls, patient_dataset):
        """ Return set if RTSTRUCT in data, else raises AttributeError """
        rt_structures = patient_dataset[DICOMModality.RTSTRUCT]
        if len(rt_structures) == 1:
            return rt_structures[0]
        else:
            #return patient_dataset[DICOMModality.RTSTRUCT][0]
            raise AttributeError("Patient data has {} RTSTRUCT sets (not 1) in "
                                 "the dose group".format(len(rt_structures)))

    @classmethod
    def get_roi_info(cls, structure_dataset):
        """ Map ROI numbers with contour names and display colour """
        roi_info = {}
        for contour_sequence in structure_dataset.ROIContourSequence:
            id = contour_sequence.ReferencedROINumber
            info = {'name': cls.get_contour_name(structure_dataset, id),
                    'RGB': cls.get_contour_color(structure_dataset, id)}
            roi_info[id] = info
        return roi_info

    @classmethod
    def get_contour_name(cls, structure_data, contour_id):
        for roi_sequence in structure_data.StructureSetROISequence:
            if contour_id == roi_sequence.ROINumber:
                return roi_sequence.ROIName

    @classmethod
    def get_contour_color(cls, structure_data, contour_id):
        for contour_sequence in structure_data.ROIContourSequence:
            if contour_id == contour_sequence.ReferencedROINumber:
                return contour_sequence.ROIDisplayColor

    @classmethod
    def get_contour_sequence(cls, structure_data, contour_id):
        return_val = None
        for contour_sequence in structure_data.ROIContourSequence:
            if contour_id == contour_sequence.ReferencedROINumber:
                return_val = contour_sequence

        if return_val is not None: return return_val
        else:
            raise ValueError("No Contour with ID {}".format(contour_id))


class RTPlanReader(DatasetReader):
    """ Class for extracting data (beam, fraction) from RTPLAN data """

    @classmethod
    def get_rt_plan_data(cls, patient_dataset):
        data = patient_dataset[DICOMModality.RTPLAN]
        return data


class RTDoseReader(DatasetReader):
    """ Class for analyzing and modifying RTDOSE data """

    @classmethod
    def get_dose_matrix(cls, patient_dataset):
        """
        Returns the summed RTDOSE Datasets in the dataset for a given DoseGroup
        """
        dose_data = patient_dataset[DICOMModality.RTDOSE]
        dose_group_sum = cls.sum_doses(dose_data)
        return dose_group_sum

    @classmethod
    def sum_doses(cls, dose_data):
        """ Sums all the doses in the list of RTDOSES for the patient """
        num_doses = len(dose_data)

        if num_doses <= 1:
            grid = DoseGrid(dose_data[0])
            cls.check_dose_units(grid)
            return DoseMatrix(grid)

        """ If > 1 RTDose dataset in dose_data: Sum all DoseGrids"""
        # Check dimensions of first element in dose_data is not (#slicesx1x1)
        if cls.dose_dimension_is_1x1(DoseGrid(dose_data[0])):
            i = 1
        else:
            i = 0

        # Initialize first DoseMatrix and increment i to add next matrix in loop
        grid = DoseGrid(dose_data[i])
        total_matrix = DoseMatrix(grid)
        i += 1

        # Add the remaining RTDose datasets
        while i < len(dose_data):
            grid = DoseGrid(dose_data[i])
            i += 1

            cls.check_dose_units(grid)
            if cls.dose_dimension_is_1x1(grid):
                continue

            matrix = DoseMatrix(grid)
            logger.debug("Dose file min: {} Gy, max: {} Gy".format(
                matrix.min_dose, matrix.max_dose))

            total_matrix.add(matrix, grid.ds.SOPInstanceUID)

        return total_matrix

    @classmethod
    def check_dose_units(cls, grid):
        if grid.ds.DoseUnits.lower() != 'gy':
            print("SOP ID\t{}\tDose units\t{}".format(
                grid.ds.SOPInstanceUID, grid.ds.DoseUnits))
            raise ValueError("Dose units of SOP UID {} is {}, not Gy".format(
                grid.ds.SOPInstanceUID, grid.ds.DoseUnits))

        if hasattr(grid.ds, 'NormalizationPoint'):
                print("Normalization point\t{}".format(
                    grid.ds.NormalizationPoint))

    @classmethod
    def dose_dimension_is_1x1(cls, grid):
        if (grid.shape[0] == 1 or grid.shape[1] == 1):
            return True
        return False
