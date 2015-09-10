import covarianceMatrix_numba as covarianceMatrix

class TomoAO(self):
    """
    Class to create tomographic covariance matrices for Shack-Hartmann WFSs

    Parameters:
        nGS (int): Number of Guide stars
        subapMask (ndarray): A 2-d or 3-d Array, 1 where sub-ap active, 0 otherwise. One mask can be given for each WFS, or if 1 given, the same is assumed for each WFS
        subapSize (ndarray): Diameter of each sub-aperture. If 1 value given, assumed all WFS have same sized sub-apertures, if array, then size for each WFS.
    """

    def __init__(self, nGS, subapMask, subapSize):

        self.nGS = nGS

        # If
        if type(subapSize)!=numpy.ndarray:
            self.subapSizes = numpy.array([subapSize]*self.nGS)
        else:
            self.subapSizes = subapSize

        if subapMask.ndim==2:
            self.subapMasks = numpy.array([subapMask]*self.nGS)
        else:
            self.subapMasks = subapMask

        self.nSubaps = self.subapMasks.sum(0)
        self.totalSubaps = self.self.sumMasks.sum()


    def calcSubapPositions(self):
        """
        Calculates the coordinates of each GS sub-aperture
        """
        self.gsPositions = numpy.zeros((2, self.totalSubaps))
        for gs in range(self.nGS):
            mask = self.subapMasks[gs]
            subapDiam = self.subapSizes[gs]
            iSubap = 0 # Counter for active sub-aps
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if mask[x,y]==1:
                        # X value
                        xPos = x*subapDiam
                        self.gsPositions[0, iSubap] = xPos
                        # Y value
                        yPos = y*subapDiam
                        self.gsPositions[1, iSubap] = yPos

        return self.gsPositions
