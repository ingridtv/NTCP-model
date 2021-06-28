"""
Created on 27/10/2020
@author: ingridtveten

TODO: Docstring
"""

import numpy as np
import logging

from constants import NUM_DECIMALS_SLICE_LOCATION
from utilities.util import find_centroid_2d, cartesian_to_polar, \
    polar_to_cartesian


logger = logging.getLogger()


class ROIStructure(object):
    """
    Class for containing structure/ROI points and references

    Attributes
    ----------
    name: str
        The ROI name
    roi_number: int
        The ROI ID number
    color: ndarray (1D, of length 3)
        [R, G, B] values for ROI color
    slice_list: ndarray (1D)
        List of ROISlices
    """

    name = None
    roi_number = None
    color = None
    slice_list = None

    def __init__(self, name, roi_number, color, contour_data):
        """ Constructor for ROIStructure """
        self.name = name
        self.roi_number = roi_number
        self.set_color(color)
        self.read_slice_data(contour_data)
        self.sort_slices()


    def __str__(self):
        info = f"ROIStructure( ID: {self.roi_number}, Name: {self.name}, " \
               f"Color: {self.color}, No slices: {len(self.slice_list)} )"
        return info

    @property
    def points_list_3d(self):
        """ Returns contour points (all slices) as list of tuples (x, y, z) """
        pts = []
        for slice in self.slice_list:
            for p in slice.points_list_3d():
                pts.append(p)
        return pts


    def set_color(self, color):
        #self.color = []
        #for c in color:
        #    if c < 1: self.color = color
        #    elif (0 <= c) and (c <= 255):
        self.color = np.asarray([float(elem)/255 for elem in color])


    def read_slice_data(self, contour_data):
        """ Iterate through the slice sequence in the contour set """
        self.slice_locations = []
        self.slice_list = []
        for contour_slice in contour_data.ContourSequence:
            roi_slice = ROISlice(self, contour_slice)
            # Found instances of z-values up to 10 decimals, causing errors
            z_location = np.round(roi_slice.slice_location,
                                  NUM_DECIMALS_SLICE_LOCATION)
            self.slice_locations.append(z_location)
            self.slice_list.append(roi_slice)

        self.slice_locations = np.sort(self.slice_locations)
        try:  # calculate slice thickness
            dz = self.slice_locations[1] - self.slice_locations[0]
            self.slice_thickness = abs(dz)
        except:
            self.slice_thickness = 0.0


    def sort_slices(self):
        """ Sort slices by slice_location """
        def slice_location(slice): return slice.slice_location
        self.slice_list.sort(key=slice_location)


    def get_slice(self, slice_location):
        """ Return slice by location """
        for slice in self.slice_list:
            if (np.round(slice.slice_location, NUM_DECIMALS_SLICE_LOCATION)
                    == np.round(slice_location, NUM_DECIMALS_SLICE_LOCATION)):
                return slice
        return None


    def get_slice_number(self, slice_number):
        """ Return slice by slice number (slice_list is sorted) """
        return self.slice_list[slice_number]


    def has_location(self, slice_location):
        return (slice_location in self.slice_locations)


    def to_convex_hull(self):
        """
        TODO: Docstring
        """
        from copy import deepcopy

        new_roi = deepcopy(self)
        for idx, roi_slice in enumerate(self.slice_list):
            new_slice = roi_slice.to_convex_hull()
            new_roi.slice_list[idx] = new_slice

        return new_roi


    def volume_convex_hull(self):
        """
        Get volume (in cm3) of the ConvexHull of the ROIStructure.

        Parameters
        ----------
        px_spacing : size-3 ndarray
            Pixel spacing in (z, x, y) directions of the ROIStructure

        Returns
        -------
        volume : float
            Volume of the ConvexHull of the ROIStructure in cm3
        """
        volume = 0
        z_spacing_cm = self.slice_thickness / 10.0
        for idx, roi_slice in enumerate(self.slice_list):
            area = roi_slice.area_convex_hull()
            area_cm2 = area/100.0     # from mm2 to cm2
            volume += (area_cm2 * z_spacing_cm)
        return volume


    def expand_contours(self, margin=1, mode='absolute'):
        """

        Parameters
        ----------
        margin : float
            Amount which the contour should be expanded by. If margin < 0, the
            contour is shrunk. Defaults to 1.
        mode : string
            How the amount of expansion should be measured. The string should
            be 'absolute' or 'relative'.
            - If 'absolute', margin is the absolute distance (in
              ROISlice/patient coordinates) to expand the contour with.
            - If 'relative', margin is the relative amount the contour should
              be expanded with (e.g. margin=1.1 expands the contour by 10 %).
        """

        from copy import deepcopy

        for idx, roi_slice in enumerate(self.slice_list):
            new_slice = deepcopy(roi_slice)
            new_slice.expand_contour(margin, mode=mode)
            self.slice_list[idx] = new_slice



class ROISlice(object):
    """
    Class for containing ROI points and information for a slice

    Attributes
    ----------
    name: str
        The ROI name
    roi_number: int
        The ROI ID number
    color: ndarray (1D, of length 3)
        [R, G, B] values for ROI color
    slice_location: float
        The slice location in the z-direction
    x, y: ndarray (1D)
        Pairs (x[i], y[i]) indicates a ROI point in the slice
    image_position_patient: ndarray (1D, of length 3)
        Position (in patient coordinates) of the first (top left) pixel
    pixel_spacing: ndarray (1D, of length 3)
        Pixel spacing in the (x, y, z) direections
    """

    name = None
    roi_number = None
    color = None
    x = None
    y = None
    slice_location = None
    referenced_ct_uid = None


    def __init__(self, parent_roi_structure, contour_slice):
        """ Constructor for ROISlice

        Parameters:
        ----------
        parent_roi_structure: ROIStructure
            The parent ROIStructure (used for reading name, id, color)
        contour_slice: RTSTRUCT Dataset > ROIContourSequence > ContourSequence
            The slice containing contour points and referenced images
        """
        self.name = parent_roi_structure.name
        self.roi_number = parent_roi_structure.roi_number
        self.color = parent_roi_structure.color
        self.referenced_ct_uid = \
            contour_slice.ContourImageSequence[0].ReferencedSOPInstanceUID

        # Get slice location, contour data as [x0, y0, z0, x1, y1, z1, ...]
        self.slice_location = np.round(contour_slice.ContourData[2],
                              NUM_DECIMALS_SLICE_LOCATION)
        self.read_contour_points(contour_slice)


    def read_contour_points(self, contour_slice):
        """ Iterate through contour data to get (x, y) points """
        for idx, name in enumerate(['x', 'y']):
            pts = contour_slice.ContourData[idx::3]
            pts.append(pts[0])  # Make sure contour closed
            pts = np.asarray([float(elem) for elem in pts])
            self.__setattr__(name, pts)


    @property
    def points_list(self):
        """ Returns the contour points as list of tuples (x, y) """
        pts_list = [(x, y) for x, y in zip(self.x, self.y)]
        return pts_list


    def points_list_3d(self):
        """ Returns the contour points as list of tuples (x, y, z) """
        z = self.slice_location
        pts_list_3d = [(x, y, z) for x, y in zip(self.x, self.y)]
        return pts_list_3d


    def set_points_from_list(self, pts):
        """ Set .x and .y list of coordinates from list of tuples (x, y) """
        x = []
        y = []
        for p in pts:
            x.append(p[0])
            y.append(p[1])
        self.x = np.asarray(x)
        self.y = np.asarray(y)


    def to_convex_hull(self):
        """
        Returns a new ROISlice where the points of the slice are the vertices
            forming the convex hull of the previous slice
        """
        from scipy.spatial import ConvexHull
        from copy import deepcopy

        pts = self.points_list
        conv_hull = ConvexHull(pts)
        # vertices = indices of points contained in the hull, c-clockwise order
        vertices = conv_hull.vertices

        x = []
        y = []
        for idx in vertices:
            x.append(pts[idx][0])
            y.append(pts[idx][1])
        # Close contours by appending first point
        x.append(x[0])
        y.append(y[0])

        new_slice = deepcopy(self)  # deepcopy to avoid in-place modifications
        new_slice.x = np.asarray(x)
        new_slice.y = np.asarray(y)
        return new_slice


    def area_convex_hull(self):
        from scipy.spatial import ConvexHull

        pts = self.points_list
        conv_hull = ConvexHull(pts)
        area = conv_hull.volume  # area = 2D volume, in patient coords/mm3
        return area


    def expand_contour(self, margin, mode='absolute'):
        """

        Parameters
        ----------
        margin : float
            Amount which the contour should be expanded by. If margin < 0, the
            contour is shrunk.
        mode : string
            How the amount of expansion should be measured. The string should
            be 'absolute' or 'relative'.
            - If 'absolute', margin is the absolute distance (in
              ROISlice/patient coordinates) to expand the contour with.
            - If 'relative', margin is the relative amount the contour should
              be expanded with (e.g. margin=1.1 expands the contour by 10 %).

        Notes
        -----
        Algorithm is as follows:
        1) Get points in convex hull
        2) Find centroid
        3) Translate points by moving centroid to origin
        4) Get polar coordinates
        5) Reformat convex hull vertices according to 'margin' and 'mode'
        6) Get cartesian coordinates
        7) Translate centroid back to original position

        TODO: Add default setting to 'no change', i.e. make relative option be
            rho = rho * (1 + margin)
          so that margin=0 (default) dose not change the value of rho, and
          otherwise specifies the amount in % to increase the radius with.
          E.g. margin=0.1 ==> 'increase rho/radius by 10 %'
          Note: Remember to update docstring
        """

        # Get points in contour and find centroid
        points_xy = self.points_list
        offset_x, offset_y = find_centroid_2d(points_xy)

        # Translate points by moving centroid to origin
        translated_pts = translate_points(points_xy, -offset_x, -offset_y)
        transformed_pts = []

        for p in translated_pts:
            # Get polar coordinates
            rho, theta = cartesian_to_polar(*p)

            # Transform vertices according to 'mode' and 'margin'
            if mode == 'absolute':
                rho = rho + margin
            elif mode == 'relative':
                rho = rho * margin
            else:
                print(f"Warning: Parameter `mode` in ROISlice.expand_contour()"
                      f"is '{mode}', and not one of 'absolute', 'relative'.")

            # Check whether point passes origin when margin < 0. That should
            # not be possible, so minimum value of rho coordinate is 0.
            rho = max(0, rho)

            # Transform back to cartesian coordinates
            x, y = polar_to_cartesian(rho, theta)
            transformed_pts.append((x, y))

        # Finally, translate points by moving centroid to original position
        transformed_pts = translate_points(transformed_pts, offset_x, offset_y)
        self.set_points_from_list(transformed_pts)


def translate_points(pts, dx, dy):
    """
    Parameters
    ----------
    pts : list, array-like
        List of points (x, y) to translate
    dx : float
        Translate by `dx` in the x direction
    dy : float
        Translate by `dy` in the y direction
    """
    new_pts = []
    for p in pts:
        translated_pt = (p[0] + dx, p[1] + dy)
        new_pts.append(translated_pt)
    return new_pts


def translate_points_3d(pts, dx, dy, dz):
    """
    Parameters
    ----------
    pts : list, array-like
        List of points (x, y, z) to translate
    dx : float
        Translate by `dx` in the x direction
    dy : float
        Translate by `dy` in the y direction
    dz : float
        Translate by `dz` in the z direction
    """
    new_pts = []
    for p in pts:
        translated_pt = (p[0] + dx, p[1] + dy, p[2] + dz)
        new_pts.append(translated_pt)
    return new_pts

