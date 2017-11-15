import numpy
import aotools
import TomoAO
import util.tel
import base.readConfig
c=base.readConfig.AOXml("ltaoSim/params.py")
# AO Parameters
TEL_DIAM = c.getVal("telDiam")
OBS = c.getVal("telSec")/TEL_DIAM

NWFS = c.getVal("nngs")+1
NXSUBAPS = numpy.array([c.getVal("wfs_nsubx")]*NWFS)
SUBAPDIAM = TEL_DIAM/NXSUBAPS
GSALT = numpy.array([0]*NWFS)#height of guide stars (0 for NGS)

wfsOverview=c.getVal("wfsOverview")
atmosGeom=c.getVal("atmosGeom")
ngsList=atmosGeom.makeNGSList("recon")

GSPOS=numpy.zeros((4,2),numpy.float64)#x, y position in radians.
for i in range(len(ngsList)):
    ngs=ngsList[i]
    theta=ngs.theta
    phi=ngs.phi
    #theta=i/(NWFS-1.)*360#equally spaced
    #phi=10.#10 arcsec off axis
    GSPOS[i,0]=phi*numpy.cos(theta)*numpy.pi/(180*3600.)
    GSPOS[i,0]=phi*numpy.sin(theta)*numpy.pi/(180*3600.)
GSPOS[NWFS-1]=0,0#on-axis
    
GSTYPE = numpy.array([1]*NWFS) # all NGS
PUPILSHIFT = numpy.array([0]*NWFS)
PUPILMAG = NXSUBAPS.copy()#numpy.array([7]*NWFS)#in units of subaps.
PUPILROT = numpy.array([0]*NWFS)

NLAYERS = c.getVal("nlayer")
LAYERHEIGHTS = c.getVal("hList")#numpy.array([0.,12376.])
CN2 = c.getVal("r0")...numpy.array([0.3,0.5])**(-5/3.)#r0**(-5./3)#numpy.array([1e-15,1e-15])
L0 = (xc.getVal("L0")]*NLAYERS#numpy.array([100.,100.])

PUPIL_MASK = util.tel.Pupil(NXSUBAPS[0],NXSUBAPS[0]/2.,NXSUBAPS[0]/2.*OBS/TEL_DIAM).fn.astype(numpy.float64)#aotools.circle(NXSUBAPS[0]/2., NXSUBAPS[0])
NSUBAPS = numpy.array([PUPIL_MASK.sum()]*NWFS)

NCPU = 1#doesn't seem to work with >1.
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
print covmat.min(),covmat.max()
