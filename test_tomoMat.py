import tomoMat
import numpy
import unittest

# Configuration parameters to use in the tests

NWFS = 1
NSUBAPS = numpy.array([4])
NXSUBAPS = numpy.array([2])
SUBAP_DIAM = numpy.array([0.5])
GSALT = numpy.array([0])
GSPOS = numpy.array([0,0]).reshape(2,1)
SUBAP_POS = numpy.array([[0, 0, 0.5, 0.5], [0, 0.5, 0, 0.5]])

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

COVMAT_RESULT = numpy.array([
    [0.0174,  0.0149,  0.0134,  0.0126,  0.0000,  0.0000,  0.0000,  -0.0014],
    [0.0149,  0.0174,  0.0126,  0.0134,  0.0000,  0.0000,  0.0014,  0.0000],
    [0.0134,  0.0126,  0.0174,  0.0149,  0.0000,  0.0014,  0.0000,  0.0000],
    [0.0126,  0.0134,  0.0149,  0.0174, -0.0014,  0.0000,  0.0000,  0.0000],
    [0.0000,  0.0000,  0.0000,  -0.0014, 0.0174,  0.0134,  0.0149,  0.0126],
    [0.0000,  0.0000,  0.0014,  0.0000,  0.0134,  0.0174,  0.0126,  0.0149],
    [0.0000,  0.0014,  0.0000,  0.0000,  0.0149,  0.0126,  0.0174,  0.0134],
   [-0.0014, 0.0000,  0.0000,  0.0000,   0.0126,  0.0149,  0.0134,  0.0174]])

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

        covxx = tomoMat.cov_XX(DU, DV, AC, AD, BC, BD, L0[0])

        self.assertTrue(numpy.allclose(covxx, COV_XX))


    def testCovYY(self):

        covyy = tomoMat.cov_YY(DU, DV, AC, AD, BC, BD, L0[0])

        self.assertTrue(numpy.allclose(covyy, COV_YY))


    def testCovXY(self):

        covxy = tomoMat.cov_XY(DU, DV, S0, L0[0])

        self.assertTrue(numpy.allclose(covxy, COV_XY))

    def testDPHI(self):
        dphi = tomoMat.DPHI(X, Y, L0[0])

        self.assertTrue(numpy.allclose(dphi, DPHI))



    def testCovMat(self):
        totElements = (2*NSUBAPS.sum())**2
        covMat = numpy.zeros(totElements, dtype="float64")

        covMat = tomoMat.matcov(
                NWFS, NSUBAPS, NXSUBAPS, SUBAP_DIAM, SUBAP_POS, GSALT, GSPOS,
                NLAYERS, LAYER_HEIGHTS, CN2, L0, covMat)

        print(numpy.array_str(covMat.reshape(NSUBAPS*2, NSUBAPS*2), precision=4))
        print(numpy.array_str(COVMAT_RESULT, precision=4))

        diff = COVMAT_RESULT - covMat

        print (abs(diff).max())
        print(numpy.array_str(diff, precision=4))
        self.assertTrue(numpy.allclose(covMat, COVMAT_RESULT, atol=0.001))
        return covMat


    def test_subapPosition(self):

        subapPos = tomoMat.subap_position(
                NWFS, NSUBAPS, NXSUBAPS, GSALT, GSPOS, SUBAP_POS, 
                NLAYERS, LAYER_HEIGHTS)
        self.assertTrue(numpy.allclose(subapPos, SUBAP_POS_RESULT))
        return subapPos



if __name__=="__main__":

    print("\nTest subap_position:")
    subapPos = test_subapPosition()
    print("subapPos:")
    print(subapPos)

    print("\nTest matcov")
    covMat = testCovMat()

    print("\n covMat:\n")
    print(numpy.array_str(covMat.reshape(NSUBAPS*2, NSUBAPS*2), precision=4))
