"""
Created on 26/10/2020
@author: ingridtveten

TODO: Docstring
"""

import numpy as np
from copy import deepcopy
import logging

from dose.dosegrid import DoseGrid
from constants import NUM_DECIMALS_SLICE_LOCATION


logger = logging.getLogger()


class DoseMatrix(object):
    """
    Class for dose matrix data and easy access to dose and coordinates

    Attributes
    ----------
    x, y, z : ndarray (1D)
        Coordinates for dose matrix voxels
    dose : ndarray (3D)
        Dose matrix (3D) containing dose data
    image_position_patient : ndarray (1D)
        Size-3 array with location (xyz) of first transmitted pixel
    pixel_spacing : ndarray (1D)
        Size-3 array with pixel spacing in (xyz) directions (in mm)
    units:  str
        Units of dose (should be 'GY', may be relative)
    masked : bool
        Indicates whether the dose matrix (dose) is masked by a ROI
    masked_by_id : int/str
        If 'masked == True', indicates which ROI ID masks the dose matrix
    masked_by : str
        If 'masked == True', indicates ROI name that masks the dose matrix
    """

    x = None
    y = None
    z = None
    dose = None
    image_position_patient = None
    pixel_spacing = None
    units = None
    masked = False
    masked_by_id = None
    masked_by = None

    def __init__(self, args=None):
        """ Constructor for DoseMatrix

        :param dose: Can be DoseGrid instance, or dict of attributes and values
        """
        if args is None: return

        if isinstance(args, DoseGrid):
            self.initialize_from_dose_grid(args)
        elif isinstance(args, DoseMatrix):
            self.initialize_by_copying(args)
        elif isinstance(args, dict):
            self.initialize_from_dict(args)
        else:
            raise AttributeError("Wrong input to DoseMatrix class")

        self.check_dimensionality()


    """-------------------------
        INIT HELPER FUNCTIONS
    -------------------------"""
    def initialize_from_dose_grid(self, dose_grid):
        self.x = dose_grid.axes[0]
        self.y = dose_grid.axes[1]
        self.z = np.round(dose_grid.axes[2], NUM_DECIMALS_SLICE_LOCATION)
        # Swap axes to patient indexing (z,y,x) rather than (x,y,z)
        self.dose = np.swapaxes(dose_grid.dose_grid, 0, 2)

        self.patient_id = dose_grid.ds.PatientID
        self.image_position_patient = dose_grid.offset  # ImagePositionPatient
        self.pixel_spacing = dose_grid.scale     # PixelSpacing in [x,y,z] dir
        self.slice_thickness = self.pixel_spacing[2]
        self.units = dose_grid.ds.DoseUnits

        logger.debug("Initializing DoseMatrix from DoseGrid with "
                    "Dose min: {},\tMax: {},\n\tUID: {}".format(
            self.min_dose, self.max_dose, dose_grid.ds.SOPInstanceUID))

    def initialize_by_copying(self, other):
        """ Copies attributes from another DoseMatrix """
        for attr in other.__dict__.keys():
            self.__setattr__(attr, other.__getattribute__(attr))

    def initialize_from_dict(self, dose_dict):
        for key in dose_dict.keys():
            self.__setattr__(key, dose_dict[key])

    @classmethod
    def copy(cls, other):
        return deepcopy(other)


    """-------------------------
        PROPERTIES
    -------------------------"""
    @property
    def min_dose(self):
        return np.min(self.dose)

    @property
    def mean_dose(self):
        return np.mean(self.dose)

    @property
    def max_dose(self):
        return np.max(self.dose)

    @property
    def shape(self):
        return np.shape(self.dose)

    @property
    def axes(self):
        return [self.x, self.y, self.z]

    #@property
    def points_xy(self):
        pts = []
        for y in self.y:
            for x in self.x:
                pts.append((x, y))
        return pts


    """-------------------------
        SIZE/SHAPE FUNCTIONS
    -------------------------"""
    def check_dimensionality(self):
        if not (np.size(self.z) == np.shape(self.dose)[0]
                and np.size(self.y) == np.shape(self.dose)[1]
                and np.size(self.x) == np.shape(self.dose)[2]):
            msg = "Mismatch dimensions of axis and dose in DoseMatrix"
            logger.error(msg)
            raise ValueError(msg)

    def is_coincident(self, other):
        if (np.size(self.x) != np.size(other.x) or
                np.size(self.y) != np.size(other.y) or
                np.size(self.z) != np.size(other.z) or
                np.shape(self.dose) != np.shape(other.dose)):
            return False
        return True

    def has_same_axes(self, other):
        '''if (self.x.any() != other.x.any()): return False
        if (self.y.any() != other.y.any()): return False
        if (self.z.any() != other.z.any()): return False'''
        if (np.array_equal(self.x, other.x) and
                np.array_equal(self.y, other.y) and
                np.array_equal(self.z, other.z)):
            return True
        return False


    def get_origin(self, order='zyx'):
        """ Get origin of the DoseMatrix

        Returns
        -------
        Origin of the DoseMatrix points in z, y, x order
        """

        if order == 'zyx':
            return [self.z[0], self.y[0], self.x[0]]
        elif order == 'xyz':
            return [self.x[0], self.y[0], self.z[0]]
        else:
            logger.warning("Warning in DoseMatrix.get_origin(). Parameter "
                           "'order' must be one of 'xyz' and 'zyx'.")

    def get_shape(self, order='zyx'):
        """ Get shape of the DoseMatrix

        Returns
        -------
        Shape of the dose matrix in z, y, x order
        """

        if order == 'zyx':
            return self.shape
        elif order == 'xyz':
            zyx = self.shape
            xyz = np.asarray([zyx[2], zyx[1], zyx[0]])
            return xyz
        else:
            logger.warning("Warning in DoseMatrix.get_shape(). Parameter "
                           "'order' must be one of 'xyz' and 'zyx'.")

    def get_spacing(self, order='zyx'):
        """ Get spacing of the DoseMatrix points

        Returns
        -------
        Spacing of the dose points in z, y, x order
        """

        if len(self.x) and len(self.y) and len(self.z):
            if order == 'zyx':
                return [self.z[1] - self.z[0],
                        self.y[1] - self.y[0],
                        self.x[1] - self.x[0]]
            elif order == 'xyz':
                return [self.x[1] - self.x[0],
                        self.y[1] - self.y[0],
                        self.z[1] - self.z[0]]
            else:
                logger.warning("Warning in DoseMatrix.get_shape(). Parameter "
                               "'order' must be one of 'xyz' and 'zyx'.")
        else:
            logger.warning("Warning: Trying to get DoseMatrix spacing, but "
                "x,y,z axes contain less than two values. Returning 'None'.")
            return None

    def translate(self, dxdydz):
        """
        Translate the DoseMatrix points by a vector dxdydz in 3D

        Parameters
        ----------
        dxdydz : array-like
            The translation vector to apply to the DoseMatrix points

        Returns
        -------
        -
        """

        shape_xyz = np.flip(self.get_shape())  # equivalent to swapping x and z
        new_axes = []
        for i, ax in enumerate(self.axes):
            pts = []
            for coord in ax:
                pts.append(coord + dxdydz[i])
            new_axes.append(pts)

        self.x = new_axes[0]
        self.y = new_axes[1]
        self.z = new_axes[2]


    """-------------------------
        GET/SET FUNCTIONS
    -------------------------"""
    def get_dose_plane(self, slice_location):
        slice_index = np.argwhere(self.z==slice_location)[0][0]
        return self.get_dose_plane_number(slice_index)

    def get_dose_plane_number(self, slice_number):
        return self.dose[slice_number]

    def set_dose_plane(self, slice_location, dose_plane):
        slice_index = np.nonzero(self.z == slice_location)[0][0]
        self.set_dose_plane_number(slice_index, dose_plane)

    def set_dose_plane_number(self, slice_number, dose_plane):
        self.dose[slice_number] = dose_plane

    def set_masking_data(self, roi_id, roi_name):
        self.masked = True
        self.masked_by_id = roi_id
        self.masked_by = roi_name


    """-------------------------
        OTHER FUNCTIONS
    -------------------------"""
    def add(self, other, uid=None):
        if uid is not None:
            logger.debug("Adding dose UID: {}".format(uid))

        if self.is_coincident(other) and self.has_same_axes(other):
            self.dose += other.dose
        elif np.max(other.dose) == 0:
            pass
        else:
            msg = "Patient ID {}: Dimensions/axes {} of DoseMatrix does not " \
                  "match dimensions/axes {} of matrix to be added".format(
                self.patient_id, np.shape(self.dose), np.shape(other.dose))
            logger.error(msg)
            raise IndexError(msg)

    def subtract(self, other):
        # Created on 2021-02-03
        # TODO: Test function

        self_copy = deepcopy(self)
        other_copy = deepcopy(other)

        if np.ma.is_masked(self_copy.dose) or np.ma.is_masked(other_copy.dose):
            # Get mask of resulting array as the union of self/other masks
            new_mask = np.logical_xor(self_copy.dose.mask, other_copy.dose.mask)
            # Flip mask since masked (invalid) elements should be 'True'
            new_mask = np.logical_not(new_mask)
            self_copy.mask = new_mask
            other_copy.mask = new_mask

        if (self_copy.is_coincident(other_copy)
                and self_copy.has_same_axes(other_copy)):
            self_copy.dose -= other_copy.dose
        else:
            msg = "Patient ID {}: Dimensions/axes {} of DoseMatrix does not " \
                  "match dimensions/axes {} of matrix to be subtracted".format(
                self.patient_id, np.shape(self.dose), np.shape(other.dose))
            logger.error(msg)
            raise IndexError(msg)

        return self_copy


