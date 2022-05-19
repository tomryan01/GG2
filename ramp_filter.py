import math
from random import sample
import numpy as np
from scipy import *
from matplotlib import pyplot as plt
import scipy

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	# omega max given by pi/scale (consequence of nyquist sampling theorem)
	#omega_max = math.pi / scale

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# compute fft of sinogram
	# axis = 1 ensures fft is in the sample direction (not angle)
	# n=m pads the result with zeros so that it matches filter dimensions
	# fftshift centres the frequencies about 0 in the center
	sin_fft = np.zeros((angles, m), dtype=complex)
	for a in range(angles):
		sin_fft[a] = scipy.fft(sinogram[a], n=m)
	fft_freq = 2*math.pi*scipy.fft.fftfreq(m)/scale
	fft_freq[0] = fft_freq[1] * (1/6)

	# get filter values by computing ramLak of m values between -omega_max and omega_max
	# then extend these values for all angles
	filter = np.full((angles, m), ramLak(fft_freq, max(abs(fft_freq)), alpha))

	# multiply filter by fourier transformed sinogram, then reshift back to standard form
	# then compute inverse fft and take absolute value
	filtered = np.multiply(filter, sin_fft)

	# no need for negative frequencies so take first n values
	#return np.absolute(scipy.fft.ifft(filtered))[:,0:n]
	ifft= np.zeros((angles, n))
	for a in range(angles):
		ifft[a] = scipy.fft.ifft(filtered[a])[0:n]
	return ifft

# compute ramLak filter value for a single omega given omega_max
def ramLak(omega, omega_max, alpha):
	return (np.absolute(omega) / (2*np.pi)) * (np.cos((np.pi/2)*(omega/omega_max))**alpha)
