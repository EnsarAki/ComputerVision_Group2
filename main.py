from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.io as sio
if __name__ == '__main__':
    mat_fname = pjoin('vidf-cvpr', 'vidf1_33_000_frame_full.mat')
    framecount = loadmat(mat_fname)
    frame = framecount['frame']
    frameloc = frame[0, 0]['loc']

    img = mpimg.imread('ucsdpeds/vidf/vidf1_33_000.y/vidf1_33_000_f001.png')
    for x in frameloc[0][0]:
        img[(math.floor(x[1])), math.floor(x[0])] = 1
        print(x[1], x[0])

    imgplot = plt.imshow(img, "gray")
    print(img.size)
    plt.show()