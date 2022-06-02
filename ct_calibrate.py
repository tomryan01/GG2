import numpy as np
import math
import scipy
from scipy import interpolate
from attenuate import attenuate
#from beam_hardening_calibrate import beam_hardening_calibrate
from ct_detect import ct_detect

def ct_calibrate(photons, material, sinogram, scale):
	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side

	# length (has to be the same as in ct_scan.py)
	try:
		n = sinogram.shape[1]
	except:
		# catch for single axis array
		n = len(sinogram)

	depths = np.linspace(0, n*scale, num=n)
	beam_calibration = ct_detect(photons, material.coeff('Water'), depths)

	# compute attenuation in air for calibration
	# 2*n*scale is the distance from emitter to detector as given in ct_scan.py
	calFactor = np.sum(attenuate(photons, material.coeff('Air'), 2*n*scale))

	# perform calibration
	sinogram = -np.log(sinogram / calFactor)
	beam_calibration = -np.log(beam_calibration / calFactor)

	# fit a polynomial that maps p -> thickness
	'''fit = np.polyfit(beam_calibration, np.linspace(0, n, num=n)*scale, 3)
	f = lambda p: fit[3] + (p*fit[2]) + ((p**2)*fit[1]) + ((p**3)*fit[0])

	# calibrate for beam hardening
	sinogram = f(sinogram)'''

	return sinogram