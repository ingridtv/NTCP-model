"""
Created on 21/05/2021
@author: ingridtveten

TODO: Description...
"""

import numpy as np


def get_image_metadata(image):
    """

    Parameters
    ----------
    image : SimpleITK.Image

    Returns
    -------
    d : dictionary
        Dictionary containing the basic metadata (geometry) about the image
    """

    d = {}
    d['origin'] = np.around(image.GetOrigin(), 3)
    d['spacing'] = np.around(image.GetSpacing(), 3)
    d['orientation'] = image.GetDirection()
    d['dimension'] = image.GetDimension()
    return d


def metadata_is_equal(metadata1, metadata2):
    is_equal = True
    for k in ['spacing', 'origin']:

        if k not in metadata1 or k not in metadata2:
            return False
        is_equal = (metadata1[k]==metadata2[k]).all()

    k = 'orientation'
    if k not in metadata1 or k not in metadata2:
        return False
    is_equal = (metadata1[k] == metadata2[k])

    is_equal = metadata1['dimension'] == metadata2['dimension']

    return is_equal


