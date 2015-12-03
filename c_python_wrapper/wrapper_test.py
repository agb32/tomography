import numpy
from aotools import circle
import TomoAO

# AO Parameters
TEL_DIAM = 4.2
OBS = 0.

NWFS = 2
NSUBAPS = numpy.array([37]*NWFS)
NXSUBAPS = numpy.array([7]*NWFS)
SUBAPDIAM = numpy.array([0.6]*NWFS)
GSALT = numpy.array([0]*NWFS)
GSPOS = numpy.array([ [.6, 0], [-0.6, 0] ]) * (1./3600) * (numpy.pi/180.)

GSTYPE = numpy.array([1]*NWFS) # all NGS
PUPILSHIFT = numpy.array([0]*NWFS)
PUPILMAG = numpy.array([1]*NWFS)
PUPILROT = numpy.array([0]*NWFS)

NLAYERS = 1
LAYERHEIGHTS = numpy.array([12376.])
CN2 = numpy.array([10.0])
L0 = numpy.array([0.5])

PUPIL_MASK = circle.circle(3.5, 7)

NCPU = 1
PART = 0


# Get the sub-aperture positions in the pupil
subapPos = (numpy.array(
        numpy.where(PUPIL_MASK==1)).T * TEL_DIAM/NXSUBAPS[0]).T
subapPos = numpy.tile(subapPos, (1,NWFS))

print(subapPos)
covMat = numpy.zeros(
        (2*NSUBAPS.sum(), 2*NSUBAPS.sum()), dtype="float64")

tomo = TomoAO.Tomo(
        NWFS, subapPos[0].copy(), subapPos[1].copy(),
        TEL_DIAM, OBS, NSUBAPS, NXSUBAPS, GSALT, GSTYPE,
        GSPOS[:, 0].copy(), GSPOS[:, 1].copy(), PUPILSHIFT,
        PUPILSHIFT, PUPILROT, PUPILMAG, SUBAPDIAM, NLAYERS,
        CN2, LAYERHEIGHTS, L0, NCPU, PART
        )

covmat = tomo.covmat(covMat)
