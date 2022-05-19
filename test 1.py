from ct_phantom import ct_phantom
from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from fake_source import fake_source
from material import Material
from ramp_filter import *
from back_project import *
from hu import *
from scan_and_reconstruct import scan_and_reconstruct
material = Material()

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

def compare_im(real_im, recon_im, threshold):
	if real_im.shape != recon_im.shape:
		raise ValueError('Reconstructed image must be the same size as the original image')
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

# Generate an ideal photon source with a single energy of 100keV filtered by 2mm of aluminuim
s = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 2, method='ideal')

#Phantom 1 - Simple circle made out of water
#The error for this is roughly 
phantom_circle = ct_phantom(material.name, 256, 1, metal='Water')
converted_cirle = conv_linatten(phantom_circle, material, 0.1)
reconstructed_circle = scan_and_reconstruct(s, material, phantom_circle, 0.01, 256)
compare_im(converted_cirle, reconstructed_circle, 1)

#Phantom 3 - Single large hip replacement made out of aluminuim
phantom_hip = ct_phantom(material.name, 256, 3, metal='Aluminium')
converted_hip = conv_linatten(phantom_hip, material, 0.1)
reconstructed_hip = scan_and_reconstruct(s, material, phantom_hip, 0.01, 256)
compare_im(converted_hip, reconstructed_hip, 1)

#Phantom 7 - Pelvic fixation pins made out of iron
phantom_fixation_pins = ct_phantom(material.name, 256, 7, metal='Iron')
converted_fixation_pins = conv_linatten(phantom_fixation_pins, material, 0.1)
reconstructed_fixation_pins = scan_and_reconstruct(s, material, phantom_fixation_pins, 0.01, 256)
compare_im(converted_fixation_pins, reconstructed_fixation_pins, 1)