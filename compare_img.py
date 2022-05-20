
from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *




def compare_im(real_im, recon_im, threshold):
	if real_im.shape != recon_im.shape:
		raise ValueError('Reconstruted image must be the same size as the origional image')
	else:
		#Find mean squared error
		error_im = np.subtract(real_im,recon_im)
		draw(error_im)
		MSE = (np.square(error_im)).mean()
		#Test if above threshold
		if MSE > threshold:
			print(MSE)
			raise ValueError('Mean squared error is above threshold')
		else:
			print("Test Passed. MSE=", MSE)