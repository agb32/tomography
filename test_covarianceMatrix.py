import covarianceMatrix
import numpy
import unittest

# Configuration parameters to use in the tests

NWFS1 = 1
NWFS2 = 2
NSUBAPS1 = numpy.array([4])
NSUBAPS2 = numpy.array([4, 4])
NXSUBAPS1 = numpy.array([2])
NXSUBAPS2 = numpy.array([2, 2])
SUBAP_DIAM1 = numpy.array([0.5])
SUBAP_DIAM2 = numpy.array([0.5,0.5])
GSALT1 = numpy.array([0])
GSALT2 = numpy.array([0, 0])
GSPOS1 = numpy.array([0,0]).reshape(1,2)
GSPOS2 = (numpy.array([[0, 10], [0, -10]])/3600.) * (numpy.pi/180.)
SUBAP_POS1 = numpy.array([[0, 0, 0.5, 0.5], [0, 0.5, 0, 0.5]])
SUBAP_POS2 = numpy.array([[0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5], [0, 0.5, 0., 0.5, 0., 0.5, 0., 0.5,]])

NLAYERS = 1
LAYER_HEIGHTS = numpy.array([5000])
CN2 = numpy.array([10.])
L0 = numpy.array([100.])

# Test values and results which have been run in the C module from LESIA
DU = 10
DV = 0.1
AC = 1
AD = 2
BC = 3.
BD = 1.
S0 = 1.
X = 1.
Y = 1.

COV_XX = 34.00398409
COV_YY = 6.69723355
COV_XY = -0.05720886
DPHI = 7.87034768

COVMAT_RESULT1 = numpy.array([
    [0.0174,  0.0149,  0.0134,  0.0126,  0.0000,  0.0000,  0.0000,  -0.0014],
    [0.0149,  0.0174,  0.0126,  0.0134,  0.0000,  0.0000,  0.0014,  0.0000],
    [0.0134,  0.0126,  0.0174,  0.0149,  0.0000,  0.0014,  0.0000,  0.0000],
    [0.0126,  0.0134,  0.0149,  0.0174, -0.0014,  0.0000,  0.0000,  0.0000],
    [0.0000,  0.0000,  0.0000,  -0.0014, 0.0174,  0.0134,  0.0149,  0.0126],
    [0.0000,  0.0000,  0.0014,  0.0000,  0.0134,  0.0174,  0.0126,  0.0149],
    [0.0000,  0.0014,  0.0000,  0.0000,  0.0149,  0.0126,  0.0174,  0.0134],
   [-0.0014, 0.0000,  0.0000,  0.0000,   0.0126,  0.0149,  0.0134,  0.0174]])

COVMAT_RESULT2 = numpy.array([
[0.0174,        0.0149,         0.0134,         0.0126,         0.0000,         0.0000,         0.0000,         -0.0014,        0.0125,         0.0150,         0.0113, 0.0127,         0.0000,         0.0000,         0.0018,         0.0013,         ],
[0.0149,        0.0174,         0.0126,         0.0134,         0.0000,         0.0000,         0.0014,         0.0000,         0.0106,         0.0125,         0.0100, 0.0113,         0.0000,         0.0000,         0.0014,         0.0018,         ],
[0.0134,        0.0126,         0.0174,         0.0149,         0.0000,         0.0014,         0.0000,         0.0000,         0.0113,         0.0127,         0.0125, 0.0150,         -0.0018,        -0.0013,        0.0000,         0.0000,         ],
[0.0126,        0.0134,         0.0149,         0.0174,         -0.0014,        0.0000,         0.0000,         0.0000,         0.0100,         0.0113,         0.0106, 0.0125,         -0.0014,        -0.0018,        0.0000,         0.0000,         ],
[0.0000,        0.0000,         0.0000,         -0.0014,        0.0174,         0.0134,         0.0149,         0.0126,         0.0000,         0.0000,         0.0018, 0.0013,         0.0081,         0.0137,         0.0088,         0.0128,         ],
[0.0000,        0.0000,         0.0014,         0.0000,         0.0134,         0.0174,         0.0126,         0.0149,         0.0000,         0.0000,         0.0014, 0.0018,         0.0057,         0.0081,         0.0062,         0.0088,         ],
[0.0000,        0.0014,         0.0000,         0.0000,         0.0149,         0.0126,         0.0174,         0.0134,         -0.0018,        -0.0013,        0.0000, 0.0000,         0.0088,         0.0128,         0.0081,         0.0137,         ],
[-0.0014,       0.0000,         0.0000,         0.0000,         0.0126,         0.0149,         0.0134,         0.0174,         -0.0014,        -0.0018,        0.0000, 0.0000,         0.0062,         0.0088,         0.0057,         0.0081,         ],
[0.0125,        0.0106,         0.0113,         0.0100,         0.0000,         0.0000,         -0.0018,        -0.0014,        0.0174,         0.0149,         0.0134, 0.0126,         0.0000,         0.0000,         0.0000,         -0.0014,        ],
[0.0150,        0.0125,         0.0127,         0.0113,         0.0000,         0.0000,         -0.0013,        -0.0018,        0.0149,         0.0174,         0.0126, 0.0134,         0.0000,         0.0000,         0.0014,         0.0000,         ],
[0.0113,        0.0100,         0.0125,         0.0106,         0.0018,         0.0014,         0.0000,         0.0000,         0.0134,         0.0126,         0.0174, 0.0149,         0.0000,         0.0014,         0.0000,         0.0000,         ],
[0.0127,        0.0113,         0.0150,         0.0125,         0.0013,         0.0018,         0.0000,         0.0000,         0.0126,         0.0134,         0.0149, 0.0174,         -0.0014,        0.0000,         0.0000,         0.0000,         ],
[0.0000,        0.0000,         -0.0018,        -0.0014,        0.0081,         0.0057,         0.0088,         0.0062,         0.0000,         0.0000,         0.0000, -0.0014,        0.0174,         0.0134,         0.0149,         0.0126,         ],
[0.0000,        0.0000,         -0.0013,        -0.0018,        0.0137,         0.0081,         0.0128,         0.0088,         0.0000,         0.0000,         0.0014, 0.0000,         0.0134,         0.0174,         0.0126,         0.0149,         ],
[0.0018,        0.0014,         0.0000,         0.0000,         0.0088,         0.0062,         0.0081,         0.0057,         0.0000,         0.0014,         0.0000, 0.0000,         0.0149,         0.0126,         0.0174,         0.0134,         ],
[0.0013,        0.0018,         0.0000,         0.0000,         0.0128,         0.0088,         0.0137,         0.0081,         -0.0014,        0.0000,         0.0000, 0.0000,         0.0126,         0.0149,         0.0134,         0.0174,         ]])

SUBAP_POS_RESULT = numpy.array([[[[ 0.  ],
         [ 0.  ],
         [ 0.25],
         [ 0.25]]],


       [[[ 0.  ],
         [ 0.25],
         [ 0.  ],
         [ 0.25]]]])

class TestCovMat(unittest.TestCase):

    def testCovXX(self):

        covxx = covarianceMatrix.cov_XX(DU, DV, AC, AD, BC, BD, L0[0])

        self.assertTrue(numpy.allclose(covxx, COV_XX))


    def testCovYY(self):

        covyy = covarianceMatrix.cov_YY(DU, DV, AC, AD, BC, BD, L0[0])

        self.assertTrue(numpy.allclose(covyy, COV_YY))


    def testCovXY(self):

        covxy = covarianceMatrix.cov_XY(DU, DV, S0, L0[0])

        self.assertTrue(numpy.allclose(covxy, COV_XY))

    def testDPHI(self):
        dphi = covarianceMatrix.DPHI(X, Y, L0[0])

        self.assertTrue(numpy.allclose(dphi, DPHI))



    def testCovMat1(self):
        covMat = numpy.zeros(
                (2*NSUBAPS1.sum(), 2*NSUBAPS1.sum()), dtype="float64")

        covMat = covarianceMatrix.matcov(
                NWFS1, NSUBAPS1, NXSUBAPS1, SUBAP_DIAM1, SUBAP_POS1, GSALT1, GSPOS1,
                NLAYERS, LAYER_HEIGHTS, CN2, L0, covMat)


        self.assertTrue(numpy.allclose(covMat, COVMAT_RESULT1, atol=0.001))
        return covMat
    
    def testCovMat2(self):
        covMat = numpy.zeros((2*NSUBAPS2.sum(), 2*NSUBAPS2.sum()),dtype="float64")

        covMat = covarianceMatrix.matcov(
                NWFS2, NSUBAPS2, NXSUBAPS2, SUBAP_DIAM2, SUBAP_POS2, GSALT2, GSPOS2,
                NLAYERS, LAYER_HEIGHTS, CN2, L0, covMat)


        self.assertTrue(numpy.allclose(covMat, COVMAT_RESULT2, atol=0.001))
        return covMat

    def test_subapPosition(self):

        subapPos = covarianceMatrix.subap_position(
                NWFS1, NSUBAPS1, NXSUBAPS1, GSALT1, GSPOS1, SUBAP_POS1, 
                NLAYERS, LAYER_HEIGHTS)
        self.assertTrue(numpy.allclose(subapPos, SUBAP_POS_RESULT))
        return subapPos



