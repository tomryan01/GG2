import numpy as np
from attenuate import *
from ct_calibrate import *
from ct_detect import *
from ct_lib import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	n = reconstruction.shape[1]

	water_atten = ct_detect(p, material.coeff('Water'), np.linspace(0, 1*n*scale, n), 1)
	
	# put this through the same calibration process as the normal CT data
	calFactor = np.sum(attenuate(p, material.coeff('Air'), 2*n*scale))
	water_atten = np.log(water_atten / calFactor)
	
	# find linear attenuation coefficient of water
	mu_w, interc = np.polyfit(np.linspace(0, n*scale, n), water_atten, 1)
	mu_w = -mu_w


	# use result to convert to hounsfield units
	# limit minimum to -1024, which is normal for CT data.
	for i in range(reconstruction.shape[0]):
		for j in range(reconstruction.shape[1]):
			reconstruction[i][j] = 1000*(reconstruction[i][j] - mu_w) / mu_w

	reconstruction = np.clip(reconstruction, -1024, 3072)

	return reconstruction