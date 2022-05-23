import numpy as np
from ct_calibrate import ct_calibrate
from ct_detect import ct_detect
from matplotlib import pyplot as plt

# generate a function to calibrate for beam hardening
def beam_hardening_calibrate(p, samples, scale, material):
	# create a linear set of depths
	depths = np.linspace(0, samples*scale, num=samples)
	# get calibrated (log-domain) attenuated values for these depths
	attenuated = ct_calibrate(p, material, ct_detect(p, material.coeff('Water'), depths), scale)
	# fit a polynomial that maps p -> thickness
	fit = np.polyfit(attenuated, np.linspace(0, len(attenuated), num=samples)*scale, 3)
	# return a function that computes thickness from p
	return lambda p: fit[3] + (p*fit[2]) + ((p**2)*fit[1]) + ((p**3)*fit[0])