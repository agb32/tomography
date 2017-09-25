import numpy
import aotools
import TomoAO

# AO Parameters
TEL_DIAM = 39
OBS = 8

NWFS = 7
NSUBAPS = numpy.array([37]*NWFS)
NXSUBAPS = numpy.array([80]*NWFS)
SUBAPDIAM = TEL_DIAM/NXSUBAPS
GSALT = numpy.array([0]*NWFS)
GSPOS = numpy.array([ [.6, 0], [-0.6, 0], [.6, 0], [-0.6, 0], [.6, 0], [-0.6, 0] ,[0,0] ]) * (1./3600) * (numpy.pi/180.)

GSTYPE = numpy.array([1]*NWFS) # all NGS
PUPILSHIFT = numpy.array([0]*NWFS)
PUPILMAG = numpy.array([80]*NWFS)
PUPILROT = numpy.array([0]*NWFS)

NLAYERS = 10
LAYERHEIGHTS = numpy.linspace(0,20000, NLAYERS)
CN2 = numpy.array([10e-15]* NLAYERS)
L0 = numpy.array([100.]*NLAYERS)

PUPIL_MASK = aotools.circle(40, 80) - aotools.circle(8.2, 80)

NSUBAPS = numpy.array([PUPIL_MASK.sum()]*NWFS)

NCPU = 10
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
