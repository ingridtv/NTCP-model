"""
Created on 21/04/2021
@author: ingridtveten

TODO: Description...
"""

import SimpleITK as sitk
import numpy as np

from constants import NUM_DECIMALS_SLICE_LOCATION



# ===========================
#  Image pre-processing
# ===========================

def image_centre_to_origin(image):
    """
    Shifts the centre of an image to the origin (0, 0, 0)

    Parameters
    ----------
    image : sitk.Image

    Returns
    -------
    image : sitk.Image
        The same image with the centre of the image located at the origin
    dxdydz : array-like
        The translation in x, y, z direction
    """

    origin = image.GetOrigin()
    size = image.GetSize()
    spacing = image.GetSpacing()
    img_centre = np.multiply(np.divide(size, 2), spacing)
    image.SetOrigin(-img_centre)

    dxdydz = np.subtract(image.GetOrigin(), origin)
    return image, dxdydz


def dose_matrix_to_SimpleITK_image(dose_matrix):
    """

    Parameters
    ----------
    dose_matrix : DoseMatrix
        The DoseMatrix whose dose array should be represented as an Image

    Returns
    -------
    dose_image : SimpleITK.Image
        A 3D image representing the 3D dose distribution in the patient
    """

    dose_image = sitk.GetImageFromArray(dose_matrix.dose)
    # Image direction defaults to identity: [100 010 001]
    dose_image.SetOrigin(dose_matrix.get_origin(order='xyz'))
    dose_image.SetSpacing(dose_matrix.get_spacing(order='xyz'))
    return dose_image


# ===========================
# Contour mask from CT image
# ===========================

def get_contour_ct_mask(ct_image, roi):
    """
    Removes the points that are outside of the contour, and return a copy
        of given CT image where points outside the ROI are set to zero

    Parameters
    ----------
    ct_image : SimpleITK.Image
        CT image with origin/spacing/dimensions to use for the masking
    roi : ROIStructure
        ROIStructure with which the data should be masked

    Returns
    -------
    mask_image : SimpleITK.Image
        Mask where points where the ROI is present are marked with 1, and
        points outside the ROI are marked with 0's. The resulting image is of
        the same dimensions as the original CT image.
    """

    # NOTE: sitk.Image dimensions are ordered XYZ
    shape = ct_image.GetSize()
    origin = ct_image.GetOrigin()
    spacing = ct_image.GetSpacing()

    z_arr = [ (origin[2] + i*spacing[2]) for i in range(shape[2]) ]

    shape_zyx = [shape[2], shape[1], shape[0]]
    mask_matrix = np.zeros(shape=shape_zyx).astype(dtype=bool)

    for idx, z in enumerate(z_arr):

        if not roi.has_location(np.round(z, NUM_DECIMALS_SLICE_LOCATION)):
            continue  # If slice not in ROI, keep 0's

        # Get contour mask for the slice
        roi_slice = roi.get_slice(z)
        mask = get_contour_mask(ct_image, roi_slice)
        mask_matrix[idx] = mask

    mask_image = sitk.GetImageFromArray(mask_matrix.astype(dtype=int))
    mask_image.SetOrigin(origin)
    mask_image.SetSpacing(spacing)
    if mask_image.GetSize() != shape:
        print("Warning: Mask image dimensions does not match original image")

    mask_image = sitk.Cast(mask_image, sitk.sitkUInt8) # Mask must be sitkUInt8
    return mask_image


def get_contour_mask(ct_image, roi_slice):
    """
    Get the contour mask array from a given roi_slice (contour)

    Parameters
    ----------
    ct_image: sitk.Image
        Used for getting the dimensions and points of the mask array
    roi_slice: ROISlice
        The ROISlice with which the dose matrix should be masked

    Returns
    -------
    mask : ndarray
        The mask for the given ROI slice. Has same dimensions as the
        corresponding z-location in the CT image
    """

    from matplotlib.path import Path

    # NOTE: sitk.Image dimensions are ordered XYZ
    shape = ct_image.GetSize()
    origin = ct_image.GetOrigin()
    spacing = ct_image.GetSpacing()

    shape_zyx = [shape[2], shape[1], shape[0]]
    x_arr = [ (origin[0] + i*spacing[0]) for i in range(shape[0]) ]
    y_arr = [ (origin[1] + i*spacing[1]) for i in range(shape[1]) ]
    points_xy = []
    for y in y_arr:
        for x in x_arr:
            points_xy.append((x, y))

    contour = Path(roi_slice.points_list, closed=True)
    mask = contour.contains_points(points_xy,
                                   #radius=1,
                                   )
    mask = mask.reshape((shape_zyx[1], shape_zyx[2]))
    return mask


# ===========================
#  Skeletonization
# ===========================

def skeletonize_skimage(image_array):
    """
    Compute skeleton of a 3D structure using the skeletonize_3d function from
    skimage.morphology.

    Parameters
    ----------
    image_array : ndarray, 2D or 3D
        An array where zeros represent background pixels, and nonzero values
        are foreground

    Returns
    -------
    skeleton : ndarray
        The thinned/skeletonized image
    """

    from skimage.morphology import skeletonize_3d

    skeleton = skeletonize_3d(image_array)
    return skeleton


def skeletonize_itk(image):
    """

    Parameters
    ----------
    image : SimpleITK.Image
        A 3D binary image. 1's represent foreground, 0's represent background

    Returns
    -------
    skeleton : SimpleITK.Image
        The skeletonized image
    """

    skeletonization_filter = sitk.BinaryThinningImageFilter()
    skeleton = skeletonization_filter.Execute(image)
    return skeleton