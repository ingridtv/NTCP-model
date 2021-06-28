"""
Created on 06/11/2020
@author: ingridtveten

TODO: Description...
"""


class BaseError(Exception):
    """ Base error class for the project """

    message = None

    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        txt = self.__class__.__name__
        txt += ": {}".format(self.message)
        return txt


"""==================================
Errors related to DVH calculation
=================================="""
class DVHCalculationError(BaseError):
    pass

class HistogramError(DVHCalculationError):
    """ Errors when calculating histograms """
    pass



"""==================================
Errors related to Patient procedures
=================================="""
class PatientError(BaseError):
    pass

class FileReadingError(PatientError):
    pass

class DoseMatrixError(PatientError):
    """ Error in calculation of dose matrices """
    pass

class ROIMaskError(PatientError):
    """ Error calculating ROI mask """
    pass

class DoseMaskingError(PatientError):
    """ Error when masking dose matrix """
    pass

class ROIError(PatientError):
    """ Error in ROI calculations """
    pass

class ContourPointsError(PatientError):
    """ Error in ROI calculations """
    pass



"""===========================================
Errors related to image registration process
==========================================="""

class ImageRegistrationError(BaseError):
    pass


class ImagePreprocessingError(ImageRegistrationError):
    pass
