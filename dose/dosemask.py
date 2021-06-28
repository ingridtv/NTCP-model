"""
Created on 31/10/2020
@author: ingridtveten

TODO: Docstring
"""

import numpy as np
from time import time
import logging

from dose.dosematrix import DoseMatrix
from constants import NUM_DECIMALS_SLICE_LOCATION


logger = logging.getLogger()


def apply_roi_mask(dose_matrix, roi_mask):
    """
    Removes the dose points that are outside of the contour, and returns a copy
        of given dose matrix where points not in the ROI are set to zero

    Parameters:
    -----------
    patient: Patient
        Patient containing dose data that should be masked
    roi_number: str
        ROI number with which the data should be masked
    """

    masked_dose_matrix = DoseMatrix.copy(dose_matrix)
    masked_dose_matrix.dose = np.ma.array(data=masked_dose_matrix.dose,
                                          mask=~roi_mask)
    return masked_dose_matrix


def get_mask_matrix(dose_matrix, roi):
    """
    Removes the dose points that are outside of the contour, and returns a copy
        of given dose matrix where points not in the ROI are set to zero

    Parameters:
    -----------
    dose_matrix: DoseMatrix
        DoseMatrix that should be masked
    roi: ROIStructure
        ROIStructure with which the data should be masked
    """
    t0 = time()

    mask_matrix = np.zeros(shape=dose_matrix.shape).astype(dtype=bool)

    for idx, z in enumerate(dose_matrix.z):

        if not roi.has_location(np.round(z, NUM_DECIMALS_SLICE_LOCATION)):
            continue  # If slice not in ROI, keep 0's

        # Get contour mask for the slice
        roi_slice = roi.get_slice(z)
        mask = get_contour_mask(dose_matrix, roi_slice)
        mask_matrix[idx] = mask

    t1 = time()
    logger.debug("Getting mask for '{}' took {:.4f} s".format(roi.name, t1-t0))
    return mask_matrix


def get_contour_mask(dose_matrix, roi_slice):
    """
    Get the contour mask array from a given roi_slice (contour)

    Parameters:
    -----------
    dose_matrix: DoseMatrix
        Used for getting the dimensions of the mask array
    roi_slice: ROISlice
        The ROISlice with which the dose matrix should be masked
    """
    from matplotlib.path import Path

    contour = Path(roi_slice.points_list, closed=True)
    mask = contour.contains_points(dose_matrix.points_xy(),
                                   #radius=1,
                                   #radius=dose_matrix.pixel_spacing[0],
                                   )
    mask = mask.reshape((dose_matrix.shape[1], dose_matrix.shape[2]))
    return mask