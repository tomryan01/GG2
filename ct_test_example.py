
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

# This function converts the index of values from the phantom to linear attenuation values so that it can be
# compared with the reconstructed version
def conv_linatten(img, material, energy):
    lin_atten_img = np.zeros(img.shape)
    en_index = int(np.where(material.mev == energy)[0][0])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lin_atten_img[i][j] = material.coeffs[int(img[i][j]), en_index]

    return lin_atten_img

# This function compares the phantom image which has been converted into linear attenuation values with the 
# reconstructed image. It first calculates an error image by subtracting the images from each other and drawing the
# resulting image. Then it works out the mean sqaured error (MSE) of the error image, the test is 'passed' if 
# the MSE is below the desired threshold. When the code is run, the MSE is ~ 0.215 which is small in comparison
# to the attenuation coefficient of water at that energy
def calc_error(real_im, recon_im):
	return np.subtract(real_im,recon_im)

def calc_mse(error_im):
	return (np.square(error_im)).mean()

def calc_ssim(real_im, recon_im):
	(score, diff) = ssim(real_im, recon_im, full=True)
	return score

# remove the outer -1 section for reconstructed image
def remove_outer_circle(reconstruction):
	ns = reconstruction.shape[1]
	xi, yi = np.meshgrid(np.arange(0,ns,1) - (ns/2) + 0.5, np.arange(0,ns,1) - (ns/2) + 0.5)
	reconstruction[np.where((xi ** 2 + yi ** 2) > (ns/2)**2)] = 0
	return reconstruction

# set up variables to be used in tests
phantom1 = ct_phantom(material.name, 256, 1, metal='Water')
phantom2 = ct_phantom(material.name, 256, 3, metal='Aluminium')
phantom3 = ct_phantom(material.name, 256, 7, metal='Iron')

s = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 2, method='ideal')

reconstruction1 = remove_outer_circle(scan_and_reconstruct(s, material, phantom1, 0.01, 256))
reconstruction2 = remove_outer_circle(scan_and_reconstruct(s, material, phantom2, 0.01, 256))
reconstruction3 = remove_outer_circle(scan_and_reconstruct(s, material, phantom3, 0.01, 256))

linatten1 = conv_linatten(phantom1, material, 0.1)
linatten2 = conv_linatten(phantom2, material, 0.1)
linatten3 = conv_linatten(phantom3, material, 0.1)

threshold1 = np.max(linatten1)*0.01
threshold2 = np.max(linatten2)*0.01
threshold3 = np.max(linatten3)*0.01

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	# Test that the mean squared error between input and output for different phantoms
	# are lower than a threshold

	assert calc_mse(calc_error(reconstruction1, linatten1)) < threshold1
	assert calc_mse(calc_error(reconstruction2, linatten2)) < threshold2
	assert calc_mse(calc_error(reconstruction3, linatten3)) < threshold3
	print("passed!")

def test_2():
	# save error maps for different phantoms to be checked visually
	save_plot(calc_error(reconstruction1, linatten1), 'results', 'Circle Phantom Error')
	save_plot(calc_error(reconstruction2, linatten2), 'results', 'Hip Replacement Phantom Error')
	save_plot(calc_error(reconstruction3, linatten3), 'results', 'Pelvic Fixation Pins Error')
	print("complete!")

def test_3():
	# explain what this test is for
	threshold = 0.1
	print(calc_ssim(reconstruction1, linatten1))
	print(calc_ssim(reconstruction2, linatten2))
	print(calc_ssim(reconstruction3, linatten3))
	assert calc_ssim(reconstruction1, linatten1) > 1 - threshold
	assert calc_ssim(reconstruction2, linatten3) >  1 - threshold
	assert calc_ssim(reconstruction3, linatten3) > 1 - threshold
	print("passed!")

# Run the various tests
print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
