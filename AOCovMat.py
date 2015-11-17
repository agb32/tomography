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
            self, gsAltitudes, gsPositions, layerHeights, cn2, L0):
        """
        Computes a theoretical covariance matrix

        Parameters:
            gsAltitudes (ndarray): array of length `nGS`, of the inverse of each guide stars altitude
            gsPositions (ndarray): 2-d array with the GS positions in arc-seconds
            layerHeights (ndarray): An array of length "number of layers", with the height of each turbulent layer
            cn2 (ndarray): An array of length `nLayers` with the strength of each turbulent layer
            L0 (ndarray): An array of length 'nLayers' with the outer scale of each turbulent layer

        Returns:
            ndarray: Covariance matrix array
        """
        nLayers = len(layerHeights)

        # Convert gsPositions to radians from arcsecs
        gsPositions = gsPositions * (1./3600) * (numpy.pi/180)

        # Actually do covariance matrix calculation
        self.covmat = covarianceMatrix.matcov(
                self.nGS, self.nSubaps, self.nxSubaps, self.subapSizes,
                self.subapPositions, gsAltitudes, gsPositions, nLayers,
                layerHeights, cn2, L0, self.covmat)

        return self.covmat


    def fitToData(
            self, rawCovMat,  nLayers, gsAltitudes, gsPositions, layerHeights,
            cn2, L0, fitGsAltitudes=False, fitGsPositions=False,
            fitLayerHeights=False, fitCn2=False, fitL0=False, callback=None):

        # Check all "fit" params are arrays, if not turn them into one!
        try:
            len(fitGsAltitudes)
        except TypeError:
            fitGsAltitudes = numpy.array([fitGsAltitudes]*self.nGS)

        try:
            len(fitGsPositions)
        except TypeError:
            fitGsPositions = numpy.array([fitGsPositions]*(self.nGS*2)).reshape((self.nGS, 2))

        try:
            len(fitLayerHeights)
        except TypeError:
            fitLayerHeights = numpy.array([fitLayerHeights]*nLayers)

        try:
            len(fitCn2)
        except TypeError:
            fitCn2 = numpy.array([fitCn2]*nLayers)

        try:
            len(fitL0)
        except TypeError:
            fitL0 = numpy.array([fitL0]*nLayers)

        # Turn all params into ndarrays of python "objects"
        # Copy all the parameters so they don't get changed later
        gsAltitudes = gsAltitudes.copy().astype("object")
        gsPositions = gsPositions.copy().astype("object")
        layerHeights = layerHeights.copy().astype("object")
        cn2 = cn2.copy().astype("object")
        L0 = L0.copy().astype("object")

        # init an array to store the first guess of variables to be fitted
        guess = numpy.array([])

        # Check if parameters are to be fitted. If so, add to guess.
        # If not, set the static argument to the value
        for i, fit in enumerate(fitGsAltitudes):
            if fit==True:
                guess = numpy.append(guess, gsAltitudes[i])
                gsAltitudes[i] = None

        gsPositions = gsPositions.reshape(2*self.nGS)
        fitGsPositions = fitGsPositions.reshape(2*self.nGS)
        for i, fit in enumerate(fitGsPositions):
            if fit:
                guess = numpy.append(guess, gsPositions[i])
                gsPositions[i] = None
        gsPositions = gsPositions.reshape((self.nGS, 2))
        fitGsPositions = fitGsPositions.reshape((self.nGS, 2))

        for i, fit in enumerate(fitLayerHeights):
            if fit:
                guess = numpy.append(guess, layerHeights[i])
                layerHeights[i] = None

        print(layerHeights)
        for i, fit in enumerate(fitCn2):
            if fit:
                guess = numpy.append(guess, cn2[i])
                cn2[i] = None

        for i, fit in enumerate(fitL0):
            if fit:
                guess = numpy.append(guess, L0[i])
                L0[i] = None

        self.guess = guess

        if len(self.guess)==0:
            raise Exception("WHOA! You ain't given me anything to fit?!?!?!")

        staticArgs = (
                rawCovMat, gsAltitudes, gsPositions, layerHeights, cn2, L0,
                callback)

        print(staticArgs)
        optResult = root(
                self.covMatOpt, guess, staticArgs, method="lm")

        print(optResult)

    def covMatFromFitParams(
            self, covMatParams, gsAltitudes,  gsPositions,
            layerHeights, cn2, L0):
        """
        Generates a covariance matrix from the optimization fitting parameters.

        All values are numpy ndarrays. Parameters which are to be fitted (i.e. not kept static during optimization) should be denoted with ``None``.
        Parameters:
            covMatParams (ndarray): A 1-d array of values which will be fitted
            gsAltitudes (ndarray): A 1-d array of inverse guide star altitudes.

        """

        print("Optimization Parameters: {}".format(covMatParams))
        nLayers = len(layerHeights)
        print("GS Altitudes: {}".format(gsAltitudes))
        print("GS Positions: {}".format(gsPositions))
        print("nLayers: {}".format(nLayers))
        print("layer Heights: {}".format(layerHeights))
        print("CN^2: {}".format(cn2))
        print("L0: {}\n".format(L0))
        # Check which params are in param Buffer and which are static
        # Fill the parameter arrays with all required values

        np = 0 # counter to keep track of fitted parameters

        gsAltitudes = gsAltitudes.copy()
        for i, val in enumerate(gsAltitudes):
            if val==None:
                gsAltitudes[i] = covMatParams[np]
                np+=1

        gsPositions = gsPositions.copy().reshape((2*self.nGS))
        for i, val in enumerate(gsPositions):
            if val==None:
                gsPositions[i] = covMatParams[np]
                np+=1
        gsPositions.resize((self.nGS, 2))

        layerHeights = layerHeights.copy()
        for i, val in enumerate(layerHeights):
            if val==None:
                layerHeights[i] = covMatParams[np]
                np+=1

        cn2 = cn2.copy()
        for i, val in enumerate(cn2):
            if val==None:
                cn2[i] = covMatParams[np]
                np+=1

        L0 = L0.copy()
        for i, val in enumerate(L0):
            if val==None:
                L0[i] = covMatParams[np]
                np+=1

        print("GS Altitudes: {}".format(gsAltitudes))
        print("GS Positions: {}".format(gsPositions))
        print("nLayers: {}".format(nLayers))
        print("layer Heights: {}".format(layerHeights))
        print("CN^2: {}".format(cn2))
        print("L0: {}".format(L0))

        # Make theoretical covariance
        covMat = self.makeCovMat(
                gsAltitudes, gsPositions, layerHeights, cn2, L0)

        return covMat

    def covMatOpt(
        self, covMatParams, rawCovMat, gsAltitudes, gsPositions, layerHeights, cn2, L0, callback):

        theoCovMat = self.covMatFromFitParams(
                covMatParams, gsAltitudes, gsPositions, layerHeights, cn2, L0)
        residual = theoCovMat - rawCovMat

        print("\n***\nRMS: {}\n***\n\n".format(
                numpy.sqrt((residual**2).mean())))

        if callback:
            callback(theoCovMat)

        return (residual**2).flatten()

if __name__ == "__main__":

    mask = circle.circle(3.5, 7)

    gsPositions = numpy.array([[1., 0], [0, 0], [-1.,0]])
    gsAltitudes = numpy.array([0, 0, 0])
    nLayers = 1
    layerHeights = numpy.array([12376.])
    cn2 = numpy.array([10.])
    L0 = numpy.array([100.])

    T = TomoAO(3, mask, 0.6)
    covMat = T.makeCovMat(
            gsAltitudes, gsPositions, layerHeights, cn2, L0)


    # T.fitToData(
    #         covMat, 1, gsAltitudes, gsPositions, layerHeights, cn2, L0,
    #         fitCn2=True)
