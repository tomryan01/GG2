
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from skimage.metrics import structural_similarity as ssim #python -m pip install scikit-image
from create_dicom import *

# create object instances
material = Material()
source = Source()

# conv_linatten converts the index of values from the phantom to linear attenuation values so that it can be
# compared with the reconstructed version.

# It finds what attenuation cofficient each material corresponds to at a certain energy value from the
# mass_attenuation_coeffs spreadsheet.

# We used a photon energy of 100keV, the same energy as the ideal photon source we used later for
# the reconstruction.
def conv_linatten(img, material, energy):
    lin_atten_img = np.zeros(img.shape)
    en_index = int(np.where(material.mev == energy)[0][0])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lin_atten_img[i][j] = material.coeffs[int(img[i][j]), en_index]

    return lin_atten_img

# First calculate an error image by subtracting the recon_imag from the real_im.
def calc_error(real_im, recon_im):
	return np.subtract(real_im,recon_im)

# Then work out the mean squared error (MSE) of the reconstruction.
def calc_mse(error_im):
	return (np.square(error_im)).mean()

# calc_ssim determines the structural similarity (ssim) of the real and reconstructed image
# ranging from 0 to 1 (perfect match).
def calc_ssim(real_im, recon_im):
	(score, diff) = ssim(real_im, recon_im, full=True)
	return score

# Function to remove the outer -1 section for the reconstructed image
def remove_outer_circle(reconstruction):
	ns = reconstruction.shape[1]
	xi, yi = np.meshgrid(np.arange(0,ns,1) - (ns/2) + 0.5, np.arange(0,ns,1) - (ns/2) + 0.5)
	reconstruction[np.where((xi ** 2 + yi ** 2) > (ns/2)**2)] = 0
	return reconstruction

# Set up variables to be used in tests
phantom1 = ct_phantom(material.name, 256, 1, metal='Water')
phantom2 = ct_phantom(material.name, 256, 3, metal='Breast Tissue')
phantom3 = ct_phantom(material.name, 256, 7, metal='Adipose')

# Generate an ideal photon source for reconstruction with a single energy of 100keV
s = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 2, method='ideal')

# Remove the outer -1 section 
reconstruction1 = remove_outer_circle(scan_and_reconstruct(s, material, phantom1, 0.01, 256))
reconstruction2 = remove_outer_circle(scan_and_reconstruct(s, material, phantom2, 0.01, 256))
reconstruction3 = remove_outer_circle(scan_and_reconstruct(s, material, phantom3, 0.01, 256))

# Convert phantoms to linear attenuation coefficients
linatten1 = conv_linatten(phantom1, material, 0.1)
linatten2 = conv_linatten(phantom2, material, 0.1)
linatten3 = conv_linatten(phantom3, material, 0.1)

# Set a threshold for passing the MSE test at 1% of the maximum linear attenuation cofficient value
threshold1 = np.max(linatten1)*0.01
threshold2 = np.max(linatten2)*0.01
threshold3 = np.max(linatten3)*0.01

def test_1():
	# What is the test? That the mean squared errors of the reconstructed images for different phantoms
	# are lower than the threshold set.
	# What do we expect? That the MSE is few orders of magnitude smaller than the linear attenuation 
	# coefficients (~ 0.17 for water, breast tissue and adipose)

	assert calc_mse(calc_error(reconstruction1, linatten1)) < threshold1
	assert calc_mse(calc_error(reconstruction2, linatten2)) < threshold2
	assert calc_mse(calc_error(reconstruction3, linatten3)) < threshold3
	print("passed!")

def test_2():
	# What is the test? That the error maps contain values close to 0 if the reconstruction is
	# accurate
	# What do we expect? Most of the error will be in the non-air parts of the image and the error should be
	# small in magnitude
	save_plot(calc_error(reconstruction1, linatten1), 'results', 'Circle Phantom Error')
	save_plot(calc_error(reconstruction2, linatten2), 'results', 'Hip Replacement Phantom Error')
	save_plot(calc_error(reconstruction3, linatten3), 'results', 'Pelvic Fixation Pins Error')
	print("complete!")

def test_3():
	# What is the test? That the ssim score is greater than 0.9 to show the 2 images are geometrically similar
	# What do we expect? The reconstructions are accurate so we expect a score over 0.9, with the simplest shape
	# (the disc) having the greatest ssim 
	threshold = 0.1
	print(calc_ssim(reconstruction1, linatten1))
	print(calc_ssim(reconstruction2, linatten2))
	print(calc_ssim(reconstruction3, linatten3))
	assert calc_ssim(reconstruction1, linatten1) > 1 - threshold
	assert calc_ssim(reconstruction2, linatten3) > 1 - threshold
	assert calc_ssim(reconstruction3, linatten3) > 1 - threshold
	print("passed!")

# Run the various tests
print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
