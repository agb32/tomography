import covarianceMatrix_numba as covarianceMatrix

class TomoAO(self):
    """
    Class to create tomographic covariance matrices for Shack-Hartmann WFSs

    Parameters:
        nGS (int): Number of Guide stars
        subapMask (ndarray): A 2-d or 3-d Array, 1 where sub-ap active, 0 otherwise. One mask can be given for each WFS, or if 1 given, the same is assumed for each WFS
        subapSize (ndarray): Diameter of each sub-aperture. If 1 value given, assumed all WFS have same sized sub-apertures, if array, then size for each WFS.
    """

    def __init__(self, nGS, subapMask):



        
