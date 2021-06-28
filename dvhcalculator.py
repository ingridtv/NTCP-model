"""
Created on 28/10/2020
@author: ingridtveten

The following module is based loosely on dvhcalc.py from dicompyler. The module
    contains functions to calculate DVHs from a (masked) dose matrix and write
    the data to file.
"""
import logging
from os.path import join
import numpy as np
from dicompylercore import dvh

from constants import BOOST_DOSE

logger = logging.getLogger()



class DVHCalculator(object):
    """
    Class to calculate DVHs. Based on dvhcalc.py from dicompyler
    """

    def __init__(self):
        """Constructor for DVHCalculator"""
        pass

    @classmethod
    def differential_dvh(cls, masked_matrix, roi_name, binsize=0.01):
        """
        Calculate differential DVH in Gy for a given patient and ROI ID

        Parameters
        ----------
        patient : Patient
            Patient dataset containing ROI and dose data
        roi_id : str
            The ROI number uniquely identifying the structure
        binsize : float, optional
            Bin size (in Gy) for the histogram
        """

        # Create DVH from dose array in DoseMatrix
        return cls.differential_dvh_from_masked_array(
                    masked_matrix.dose, roi_name, binsize)

    @classmethod
    def differential_dvh_from_masked_array(cls, masked_array, roi_name,
                                           binsize=0.01):

        # Create histogram from dose points
        dose_points = masked_array.compressed()
        max_dose = np.ceil(np.max(masked_array))
        hist, edges = np.histogram(dose_points,
                                   bins=np.arange(0, max_dose + 1, binsize))

        # Generate DVH
        dvh_data = dvh.DVH(counts=hist, bins=edges,
                           dvh_type='differential',
                           name=roi_name,
                           # dose_units=abs_dose_units,
                           # volume_units=abs_volume_units,
                           rx_dose=BOOST_DOSE
                           )
        return dvh_data

    @classmethod
    def cumulative_dvh(cls, masked_matrix, roi_name, binsize=0.01):
        """
        Calculate cumulative DVH in Gy for a given patient and ROI ID

        Parameters
        ----------
        patient : Patient
            Patient dataset containing ROI and dose data
        roi_id : str
            The ROI number uniquely identifying the structure
        binsize : float, optional
            Bin size for the histogram
        """

        diff_dvh = cls.differential_dvh(masked_matrix, roi_name,
                                        binsize=binsize)
        return diff_dvh.cumulative


#================================================================
#   GET DVH AND DOSE DATA
#================================================================

def calculate_and_write_dvh_to_file(patient, struct):
    """
    Calculates differential DVH and ROI volume and writes to file. Then,
        generate cumulative DVH, print DVH stats and write voxel data to file

    Returns
    -------
    patient_id : str
        Returns patient_id if error occurs, else returns None
    """

    from utilities.exceptions import ROIError, DVHCalculationError
    from utilities.util import print_structure_stats, get_masked_dose_matrix
    from fileio.readwritedata import write_voxel_data_to_file, write_dvh_to_file

    patient_id = patient.get_id()

    try:
        # Get masked dose matrix for 'struct'
        masked_matrix, roi_name, roi_volume, roi_id = \
            get_masked_dose_matrix(patient, struct)
        logger.debug("Patient ID: {}, ROI of interest: {}, ROI name: {}".format(
            patient_id, struct, roi_name))

        # Differential, absolute volume DVH
        diff_dvh = DVHCalculator.differential_dvh(masked_matrix, roi_name)
        cum_dvh = diff_dvh.cumulative
        """write_dvh_to_file(patient_id, struct, roi_volume, diff_dvh)

        # Read voxel data and generate file from masked matrix
        write_voxel_data_to_file(
          patient, struct, masked_matrix, roi_volume, roi_id=roi_id)"""

        print_structure_stats(patient_id, roi_name, roi_volume, cum_dvh)

        # Return None if everything works
        return None

    except ROIError as err:
        print("{:<10}\tNo ROI. {}".format(
            patient_id, err))
    except ValueError as err:
        print("{:<10}\tNo ROI. {}: {}".format(
            patient_id, err.__class__.__name__, err))
    except DVHCalculationError as err:
        print("{:<10}\tNo ROI. {}: {}".format(
            patient_id, err.__class__.__name__, err))
    except AttributeError as err:
        print("{:<10}\tNo ROI. {}: {}".format(
            patient_id, err.__class__.__name__, err))

    # Returns patient_id if any of above errors occur
    return patient_id

    