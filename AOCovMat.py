import numpy
from matplotlib import pyplot
from scipy.optimize import minimize, root

import covarianceMatrix_numba as covarianceMatrix
from aotools import circle

class TomoAO(object):
    """
    Class to create tomographic covariance matrices for Shack-Hartmann WFSs

    Parameters:
        nGS (int): Number of Guide stars
        subapMask (ndarray): A 2-d or 3-d Array, 1 where sub-ap active, 0 otherwise. One mask can be given for each WFS, or if 1 given, the same is assumed for each WFS
        subapSize (ndarray): Diameter of each sub-aperture. If 1 value given, assumed all WFS have same sized sub-apertures, if array, then size for each WFS.
        GSPositions (ndarray): 2-d array of x and y GS positions in arcsecs
    """

    def __init__(self, nGS, subapMask, subapSize):

        self.nGS = nGS

        # If
        if type(subapSize)!=numpy.ndarray:
            self.subapSizes = numpy.array([subapSize]*self.nGS)
        else:
            self.subapSizes = subapSize

        if subapMask.ndim==2:
            self.subapMasks = [subapMask]*self.nGS
        else:
            self.subapMasks = subapMask

        self.nSubaps = []
        self.nxSubaps = []
        self.totalSubaps = 0
        for gs in range(self.nGS):
            self.nSubaps.append(self.subapMasks[gs].sum())
            self.nxSubaps.append(self.subapMasks[gs].shape[0])
            self.totalSubaps += self.subapMasks[gs].sum()

        self.nSubaps = numpy.array(self.nSubaps, dtype="int")
        self.nxSubaps = numpy.array(self.nxSubaps, dtype="int")

        self.subapPositions = self.calcSubapPositions()

        self.covmat = numpy.zeros(
                (2*self.totalSubaps, 2*self.totalSubaps), dtype="float32")



    def calcSubapPositions(self):
        """
        Calculates the coordinates of each GS sub-aperture at the pupil altitude
        """
        self.subapPositions = numpy.zeros((2, self.totalSubaps))
        iSubap = 0 # Counter for active sub-aps
        for gs in range(self.nGS):
            mask = self.subapMasks[gs]
            subapPos = numpy.array(numpy.where(mask==1))*self.subapSizes[gs]
            self.subapPositions[:, iSubap:iSubap+self.nSubaps[gs]] = subapPos
            iSubap+=self.nSubaps[gs]

        return self.subapPositions


    def makeCovMat(
            self, gsAltitudes, gsPositions, nLayers, layerHeights, cn2, L0):

        gsPositions = gsPositions * (1./3600) * (numpy.pi/180)

        # Actually do covariance matrix calculation
        self.covmat = covarianceMatrix.matcov(
                self.nGS, self.nSubaps, self.nxSubaps, self.subapSizes,
                self.subapPositions, gsAltitudes, gsPositions, nLayers,
                layerHeights, cn2, L0, self.covmat)

        return self.covmat


    def fitToData(
            self, rawCovMat,  nLayers, gsAltitudes, gsPositions, layerHeights,
            cn2, L0, fitGsAltitudes=True, fitGsPositions=True,
            fitLayerHeights=True, fitCn2=True, fitL0=True):
        guess = numpy.array([])

        # Check if parameters are to be fitted. If so, add to guess.
        # If not, set the static argument to the value
        if fitGsAltitudes:
            guess = numpy.append(guess, gsAltitudes.flatten())
            staticGsAltitudes = None
        else:
            staticGsAltitudes = gsAltitudes

        if fitGsPositions:
            guess = numpy.append(guess, gsPositions.flatten())
            staticGsPositions = None
        else:
            staticGsPositions = gsPositions

        if fitLayerHeights:
            guess = numpy.append(guess, layerHeights.flatten())
            staticLayerHeights = None
        else:
            staticLayerHeights = layerHeights

        if fitCn2:
            guess = numpy.append(guess, cn2.flatten())
            staticCn2 = None
        else:
            staticCn2 = cn2

        if fitL0:
            guess = numpy.append(guess, L0.flatten())
            staticL0 = None
        else:
            staticL0 = L0

        self.guess = guess

        staticArgs = (  rawCovMat, self.nGS, nLayers, staticGsAltitudes,
                        staticGsPositions, staticLayerHeights, staticCn2,
                        staticL0)

        optResult = root(self.getFitError, guess, staticArgs, method="lm")

        print(optResult)

    def covMatFromParamBuf(
            self, covMatParams, nGS, nLayers, gsPositions=None,
            gsAltitudes=None, layerHeights=None, cn2=None, L0=None):


        # Check which params are in param Buffer and which are static
        i = 0
        if gsPositions==None:
            gsPositions = covMatParams[i:i+(2*nGS)].reshape(nGS, 2)
            i+=(2*nGS)

        print(gsAltitudes)
        if gsAltitudes==None:
            gsAltitudes = covMatParams[i:i+nGS]
            i+=nGS

        if layerHeights==None:
            layerHeights = covMatParams[i:i+nLayers]
            i+=nLayers

        if cn2==None:
            cn2 = covMatParams[i:i+nLayers]
            i+=nLayers

        if L0==None:
            L0 = covMatParams[i:i+nLayers]
            i+=nLayers

        print("GS Altitudes: {}".format(gsAltitudes))
        print("GS Positions: {}".format(gsPositions))
        print("nLayers: {}".format(nLayers))
        print("layer Heights: {}".format(layerHeights))
        print("CN^2: {}".format(cn2))
        print("L0: {}".format(L0))

        # Make theoretical covariance
        covMat = self.makeCovMat(
                gsAltitudes, gsPositions, nLayers, layerHeights, cn2, L0)

        return covMat

    def getFitError(
        self, covMatParams, rawCovMat, nGS, nLayers, gsPositions=None,
        gsAltitudes=None, layerHeights=None, cn2=None, L0=None):


        theoCovMat = self.covMatFromParamBuf(
                covMatParams, nGS, nLayers, gsPositions, gsAltitudes,
                layerHeights, cn2, L0)
        residual = theoCovMat - rawCovMat

        print("\n***\nRMS: {}\n***\n".format(numpy.sqrt((residual**2).mean())))

        return (residual**2).flatten()

if __name__ == "__main__":

    mask = circle.circle(3.5, 7)

    gsPositions = numpy.array([[1, 0], [0, 0], [-1,0]])
    gsAltitudes = numpy.array([0, 0, 0])
    nLayers = 1
    layerHeights = numpy.array([12376.])
    cn2 = numpy.array([10.])
    L0 = numpy.array([100.])

    T = TomoAO(3, mask, 0.6)
    covMat = T.makeCovMat(
            gsAltitudes, gsPositions, nLayers, layerHeights, cn2, L0)


    pyplot.imshow(covMat, origin="lower")
    pyplot.show()
