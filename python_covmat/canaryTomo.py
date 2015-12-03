from astropy.io import fits
import numpy
import os
from matplotlib import pyplot

from aotools import circle

import covarianceMatrix_numba as covarianceMatrix


# Parameters for CANARY
TEL_DIAM = 4.2

NWFS = 2
NSUBAPS = numpy.array([37]*NWFS)
NXSUBAPS = numpy.array([7]*NWFS)
SUBAPDIAM = numpy.array([0.6]*NWFS)
GSALT = numpy.array([0]*NWFS)
GSPOS = numpy.array([ [.6, 0], [-.6, 0] ]) * (1./3600) * (numpy.pi/180.)

NLAYERS = 1
LAYERHEIGHTS = numpy.array([12376.])
CN2 = numpy.array([10.0])
L0 = numpy.array([0.5])

PUPIL_MASK = circle.circle(3.5, 7)



def canaryCovMat():
	subapPos = (numpy.array(numpy.where(PUPIL_MASK==1)).T * TEL_DIAM/NXSUBAPS[0]).T
	subapPos = numpy.tile(subapPos, (1,NWFS))

	print(subapPos)
	covMat = numpy.zeros( (2*NSUBAPS.sum(), 2*NSUBAPS.sum()), dtype="float64")

	covMat = fillCovMat(covMat, subapPos)

	return covMat

def fillCovMat(covMat, subapPos):

	covMat = covarianceMatrix.matcov(
			NWFS, NSUBAPS, NXSUBAPS, SUBAPDIAM, subapPos, GSALT, GSPOS,
			NLAYERS, LAYERHEIGHTS, CN2, L0, covMat)

	return covMat

if __name__ == "__main__":

	covMat = canaryCovMat()

	pyplot.imshow(covMat, origin="lower")
	pyplot.show()

	# PXL_SCALE = 2.5/14.
	# TOT_SUBAPS = 74
	# NX_SUBAPS = 7
	# SUBAP_DIAM = 0.6
	# GUESS = [(0.5, 0.), 0.18]

	# canary = LATomo(PXL_SCALE, TOT_SUBAPS, NX_SUBAPS, SUBAP_DIAM)
	# canary.loadRawData(os.environ["HOME"]+"/CfAI/tomography/sim/CANARY_2WFS_tomo/2015-05-13-17-03-18/slopes.fits")
	# canary.fitRawData(GUESS, plot=True, mode=2)
