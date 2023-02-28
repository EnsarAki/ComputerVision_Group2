from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal
from PIL import Image

import scipy.io as sio
if __name__ == '__main__':
    mat_fname = pjoin('vidf-cvpr', 'vidf1_33_000_frame_full.mat')
    framecount = loadmat(mat_fname)
    frame = framecount['frame']
    frameloc = frame[0, 0]['loc']

    img = mpimg.imread('ucsdpeds/vidf/vidf1_33_000.y/vidf1_33_000_f001.png')

    sigma = 5
    cov_matrix = [[sigma, 0], [0, sigma]]
    column_size = len(img[0])
    row_size = len(img)

    img_density = np.zeros((row_size, column_size))

    for n in frameloc[0][0]:
        x = math.floor(n[1])
        y = math.floor(n[0])
        mean = [x, y]
        for i in range(row_size):
            for j in range(column_size):
                img_density[i, j] = img_density[i, j] + multivariate_normal.pdf([i, j], mean, cov_matrix)

    imgplot1 = plt.imshow(img, "gray")
    plt.show()
    imgplot2 = plt.imshow(img_density)
    plt.show()