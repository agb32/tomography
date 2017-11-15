import time
import numpy
import aotools
import TomoAO

# AO Parameters
TEL_DIAM = 39.
OBS = 8.

NWFS = 7

NXSUBAPS = numpy.array([80]*NWFS).astype(numpy.int64)
SUBAPDIAM = (TEL_DIAM/NXSUBAPS).astype(numpy.float64)
GSALT = numpy.array([0.]*NWFS).astype(numpy.float32)
GSPOS = (numpy.array([ [.6, 0], [-0.6, 0], [.6, 0], [-0.6, 0], [.6, 0], [-0.6, 0] ,[0,0] ]) * (1./3600) * (numpy.pi/180.)).astype(numpy.float64)

GSTYPE = numpy.array([1]*NWFS).astype(numpy.int32) # all NGS
PUPILSHIFT = numpy.array([0]*NWFS).astype(numpy.float64)
PUPILMAG = numpy.array([80]*NWFS).astype(numpy.float64)
PUPILROT = numpy.array([0]*NWFS).astype(numpy.float64)

NLAYERS = 10
LAYERHEIGHTS = numpy.linspace(0,20000, NLAYERS).astype(numpy.float64)
CN2 = (numpy.array([10e-15]* NLAYERS)).astype(numpy.float64)
L0 = (numpy.array([100.]*NLAYERS)).astype(numpy.float64)

PUPIL_MASK = aotools.circle(40, 80) - aotools.circle(8.2, 80)

NSUBAPS = numpy.array([PUPIL_MASK.sum()]*NWFS)

NCPU = 60
PART = 0


# Get the sub-aperture positions in the pupil
subapPos = (numpy.array(
        numpy.where(PUPIL_MASK==1)).T * TEL_DIAM/NXSUBAPS[0]).T
subapPos = numpy.tile(subapPos, (1,NWFS))

print(subapPos)
covMat = numpy.zeros(
        (int(2*NSUBAPS.sum()), int(2*NSUBAPS.sum())))
print("Init TOMO Object...")
tomo = TomoAO.Tomo(
        NWFS, subapPos[0].copy(), subapPos[1].copy(),
        TEL_DIAM, OBS, NSUBAPS, NXSUBAPS, GSALT, GSTYPE,
        GSPOS[:, 0].copy(), GSPOS[:, 1].copy(), PUPILSHIFT,
        PUPILSHIFT, PUPILROT, PUPILMAG, SUBAPDIAM, NLAYERS,
        CN2, LAYERHEIGHTS, L0, NCPU, PART
        )
print("Make Covariance Matrix...")
t1 = time.time()
covmat = tomo.covmat(covMat)
dt = time.time() - t1


print("Time taken: {}s".format(dt))
