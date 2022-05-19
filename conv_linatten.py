
from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *


def conv_linatten(img, material, energy):
    lin_atten_img = np.zeros(img.shape)
    en_index = int(np.where(material.mev == energy)[0][0])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lin_atten_img[i][j] = material.coeffs[int(img[i][j]), en_index]

    return lin_atten_img