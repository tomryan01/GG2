# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from skimage.metrics import structural_similarity as ssim
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

def ssim_test():
	phantom = ct_phantom(material.name, 256, 3, 'Titanium')
	reconstruction = scan_and_reconstruct(source.photon('100kVp, 2mm Al'), material, phantom, 0.1, 256)
	(score, diff) = ssim(phantom, reconstruction, full=True)
	print("SSIM: {}".format(score))
