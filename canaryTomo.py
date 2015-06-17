from astropy.io import fits
import numpy
import os
from matplotlib import pyplot

from aotools import circle

import tomoMatrix

class LATomo(object):

	def __init__(self, pxlScale, totSubaps, nxSubaps, subapDiam):
		self.pxlScale = pxlScale
		self.totSubaps = totSubaps
		self.nxSubaps = nxSubaps
		self.subapMask = circle.circle(nxSubaps/2., nxSubaps)
		self.subapDiam = subapDiam

	def loadRawData(self, filename):
		self.rawSlopes = fits.getdata(filename)

		self.rawSlopes_rad = (self.rawSlopes * 
				self.pxlScale * numpy.pi/180./3600.)

		self.covMat_raw = numpy.cov(self.rawSlopes_rad.T)[:self.totSubaps, self.totSubaps:]

		return self.covMat_raw

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

		# if plot:
		# 	rawCov_fig = pyplot.figure()
		# 	rawCov_ax = rawCov_fig

		if mode==1:
			self.theo_covMat = tomoMatrix.fitNgsCovMat_root(
					self.covMat_raw, self.nxSubaps, self.subapDiam, 
					self.subapMask, guess_r0, guess_pos, L01=10e6, plot=plot)
		elif mode==2:
			self.theo_covMat = tomoMatrix.fitNgsCovMat_minimise(
					self.covMat_raw, self.nxSubaps, self.subapDiam, 
					self.subapMask, guess_r0, guess_pos, L01=10e6, plot=plot)

if __name__ == "__main__":

	PXL_SCALE = 2.5/14.
	TOT_SUBAPS = 74
	NX_SUBAPS = 7
	SUBAP_DIAM = 0.6
	GUESS = [(0.5, 0.), 0.18]

	canary = LATomo(PXL_SCALE, TOT_SUBAPS, NX_SUBAPS, SUBAP_DIAM)
	canary.loadRawData(os.environ["HOME"]+"/CfAI/tomography/sim/CANARY_2WFS_tomo/2015-05-13-17-03-18/slopes.fits")
	canary.fitRawData(GUESS, plot=True, mode=2)

