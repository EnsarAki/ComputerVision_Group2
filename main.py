from scipy.io import loadmat
from os.path import dirname, join as pjoin

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.io as sio
if __name__ == '__main__':
    print("hello world")
    mat_fname = pjoin('vidf-cvpr', 'vidf1_33_000_frame_full.mat')
    framecount = loadmat(mat_fname)
    frame = framecount['frame']
    print(frame[0, 0]['loc'])

    img = mpimg.imread('ucsdpeds/vidf/vidf1_33_000.y/vidf1_33_000_f001.png')
    img[96, 54] = 1
    imgplot = plt.imshow(img)
    print(img.size)
    plt.show()