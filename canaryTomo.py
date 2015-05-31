from astropy.io import fits
import numpy

from aotools import circle

import tomoMatrix

PXL_SCALE = 2.5/14.
N_SUBAPS = 74
NX_SUBAPS = 74
SUBAP_DIAM = 0.6

class LATomo(object):

	def __init__(self, pxlScale, totSubaps, nxSubaps, subapDiam):
		self.pxlScale = pxlScale
		self.nSubaps = totSubaps
		self.nxSubaps = nxSubaps
		self.subapMask = circle.circle(nxSubaps/2., nxSubaps)
		self.subapDiam = subapDiam

	def loadRawData(self, filename):
		self.rawSlopes = fits.getdata(filename)

		self.rawSlopes_rad = (self.rawSlopes * 
				self.pxlScale * numpy.pi/180./3600.)

		self.raw_covMat = numpy.cov(self.rawSlopes_raw.T)[:self.nSubaps, self.nxSubaps:]

		return self.raw_covMat

	def fitRawData(self, guess, plot=False, mode=1):
		"""
		Attempt to fit a theoretical covariance matrix to the raw data.

		Parameters:
			guess (tuple): A guess of the initial parameters, (pos, r0)
			plot (bool, optional): Plot the theoretical covariance matrices
			mode (int, optional): mode 1 uses scipy `root`, mode 2 used scipy `minimize`

		"""
		guess_r0 = guess[1]
		guess_pos = guess[0]
		if mode==1:
			self.theo_covMat = tomoMatrix.fitNgsCovMat_root(
					self.raw_covMat, self.nxSubaps, self.subapDiam, 
					self.subapMask, guess_r0, guess_pos, L0=10e6, plot=plot)
		elif mode==2:
			self.theo_covMat = tomoMatrix.fitNgsCovMat_minimise(
					self.raw_covMat, self.nxSubaps, self.subapDiam, 
					self.subapMask, guess_r0, guess_pos, L0=10e6, plot=plot)

if __name__ == "__main__":
