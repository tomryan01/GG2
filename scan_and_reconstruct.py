#from beam_hardening_calibrate import beam_hardening_calibrate
from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *

def scan_and_reconstruct(photons, material, phantom, scale, angles, mas=10000, alpha=0.001):

	""" Simulation of the CT scanning process
		reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
		takes the phantom data in phantom (samples x samples), scans it using the
		source photons and material information given, as well as the scale (in cm),
		number of angles, time-current product in mas, and raised-cosine power
		alpha for filtering. The output reconstruction is the same size as phantom."""


	# convert source (photons per (mas, cm^2)) to photons

	# create sinogram from phantom data, with received detector values
	sin = ct_scan(photons, material, phantom, scale, angles)

	# convert detector values into calibrated attenuation values
	sin = ct_calibrate(photons, material, sin, scale=scale)

	# Ram-Lak
	sin = ramp_filter(sin, scale, alpha=alpha)

	# Back-projection
	fbp = back_project(sin)

	# convert to Hounsfield Units
	fbp_hu = hu(photons, material, fbp, scale)

	return fbp_hu