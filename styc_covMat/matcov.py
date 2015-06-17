import numpy
from astropy.io import fits
import matplotlib

from aotools import circle

import matcov_cffi_header


class MatCov(object):

    def __init__(self, lib_filename="matcov.dylib"):

        self.ffiTomo, self.libTomo = matcov_cffi_header.getCovMatLib(
            lib_filename)

        self.tomoData = None

    def loadTomoData(
    		self, nWFS, subapCoordsX, subapCoordsY, telDiam, obsDiam, nSubaps, 
    		nxSubaps, gsAlt, wfsType, gsPosX, gsPosY, subapDiam, nLayers, 
    		cn2, layerHeights, L0, xPupilShift=None, yPupilShift=None,
            wfsRotation=None, wfsMagnification=None, ncpu=1, covMatPart=0
            ):

        # Set some sensible defaults for lesser used params
        if numpy.all(xPupilShift) == None:
            xPupilShift = numpy.zeros(nWFS)

        if numpy.all(yPupilShift) == None:
            yPupilShift = numpy.zeros(nWFS)

        if numpy.all(wfsRotation) == None:
            wfsRotation = numpy.zeros(nWFS)

        if wfsMagnification == None:
            wfsMagnification = numpy.ones(nWFS)

        # Remember some important params for later
        self.nWfs = nWFS
        self.nSubaps = nSubaps

        # Cast arrays to C pointers
        c_subapCoordsX = self.ffiTomo.cast(
            "double*", subapCoordsX.flatten().ctypes.data)
        c_subapCoordsY = self.ffiTomo.cast(
            "double*", subapCoordsY.flatten().ctypes.data)

        c_nSubaps = self.ffiTomo.cast(
            "long*", nSubaps.astype("int64").ctypes.data)
        c_nxSubaps = self.ffiTomo.cast(
            "long*", nSubaps.astype("int64").ctypes.data)

        c_gsAlt = self.ffiTomo.cast(
            "double*", gsAlt.astype("float64").ctypes.data)
        c_wfsType = self.ffiTomo.cast(
            "int*", wfsType.astype("int32").ctypes.data)

        c_gsPosX = self.ffiTomo.cast(
            "double*", gsPosX.astype("float64").ctypes.data)
        c_gsPosY = self.ffiTomo.cast(
            "double*", gsPosY.astype("float64").ctypes.data)

        c_xPupilShift = self.ffiTomo.cast(
            "double*", xPupilShift.astype("float64").ctypes.data)
        c_yPupilShift = self.ffiTomo.cast(
            "double*", yPupilShift.astype("float64").ctypes.data)

        c_wfsRotation = self.ffiTomo.cast(
            "double*", wfsRotation.astype("float64").ctypes.data)
        c_wfsMagnification = self.ffiTomo.cast(
            "double*", wfsMagnification.astype("float64").ctypes.data)

        c_subapDiam = self.ffiTomo.cast(
            "double*", subapDiam.astype("float64").ctypes.data)

        c_nLayers = self.ffiTomo.cast("long", nLayers)
        c_cn2 = self.ffiTomo.cast("double*", cn2.astype("float64").ctypes.data)
        c_layerHeights = self.ffiTomo.cast(
            "double*", layerHeights.astype("float64").ctypes.data)
        c_L0 = self.ffiTomo.cast("double*", L0.astype("float64").ctypes.data)

        c_ncpu = self.ffiTomo.cast("int", ncpu)
        c_covMatPart = self.ffiTomo.cast("int", covMatPart)

        # self.tomoData = self.ffiTomo.new("tomo_struct*",
        self.tomoData = [
        		long(nWFS), c_subapCoordsX, c_subapCoordsY, float(telDiam),
                float(obsDiam), c_nSubaps, c_nxSubaps, c_gsAlt, c_wfsType,
                c_gsPosX, c_gsPosY, c_xPupilShift, c_yPupilShift,
                c_wfsRotation, c_wfsMagnification, c_subapDiam, c_nLayers,
                c_cn2, c_layerHeights, c_L0, c_ncpu, c_covMatPart
                ]

    def makeCovMat(self):

        if self.tomoData == None:
            raise Exception("Load Tomo data first!")

        # Find the size of the completed covariance matrix
        covMatSize = nSubaps.sum() * 2

        # Make it, and cast to C pointer
        covMat = numpy.zeros((covMatSize,) * 2, dtype="float64")
        c_covMat = self.ffiTomo.cast("double*", covMat.ctypes.data)

        # Populate the covariance matrix
        print("Try to make CMAT with these params: {}".format(self.tomoData))
        c_tomoData = self.ffiTomo.cast("struct", self.tomoData)
        self.libTomo.matcov_styc(c_tomoData, c_covMat)

        self.covMat = numpy.frombuffer(
            self.ffiTomo.buffer(c_covMat, size=covMatSize**2),
            dtype="float64")

        return self.covMat


    def cov_XX(du, dv, ac, ad, bc, bd, L0):

        pass

if __name__ == "__main__":

    covMat = MatCov("matcov.dylib")

    xCoords, yCoords = numpy.where(circle.circle(3.5, 7))

    xCoords = numpy.tile(xCoords, 2) * 4.2
    yCoords = numpy.tile(yCoords, 2) * 4.2

    nSubaps = numpy.array([37, 37])
    nxSubaps = numpy.array([7, 7])

    gsAlt = numpy.array([0, 0])

    wfsType = numpy.array([1, 1])

    gsPosX = numpy.array([0, 5])
    gsPosY = numpy.array([0, -5])

    subapDiam = numpy.array([0.6, 0.6])

    cn2 = numpy.array([10.])
    h = numpy.array([12376])
    L0 = numpy.array([100.])

    covMat.loadTomoData(
        2, xCoords, yCoords, 4.2, 0., nSubaps, nxSubaps, gsAlt, wfsType,
        gsPosX, gsPosY, subapDiam, 1, cn2, h, L0
    )
    cMat = covMat.makeCovMat()
