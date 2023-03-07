from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy as np
import math
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal


def loadFrameLoc(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")
    # Formatting video number for loading
    vidNum = "{0:0=3d}".format(vidNum - 1)

    # Loading video locations for the frame
    mat_fname = pjoin('vidf-cvpr', f'vidf1_33_{vidNum}_frame_full.mat')
    loadedFrame = loadmat(mat_fname)

    return loadedFrame['frame'][0, (frameNum - 1)]['loc'][0][0]


def loadImage(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")

    # Formatting video number and frame number for loading
    vidNum = "{0:0=3d}".format(vidNum - 1)
    frameNum = "{0:0=3d}".format(frameNum)
    return mpimg.imread(f'ucsdpeds/vidf/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.png')


if __name__ == '__main__':
    sigma = 5
    cov_matrix = np.array([[sigma, 0], [0, sigma]])
    mean = np.zeros(2)
    mv_array = np.zeros(2)

    for vidNum in range(1, 11):
        for frameNum in range(1, 10):
            img_density = np.zeros((158, 238))
            img = loadImage(vidNum, frameNum)
            location = loadFrameLoc(vidNum, frameNum)
            for loc in location:
                mean.put([0, 1], [math.floor(loc[1]), math.floor(loc[0])])
                for i in range(158):
                    for j in range(238):
                        mv_array.put([0, 1], [i, j])
                        img_density[i, j] = img_density[i, j] + multivariate_normal.pdf(mv_array, mean, cov_matrix)
            mpimg.imsave(f'vidf-cvpr-density-map/vidf1_33_{vidNum}_f{frameNum}.png', img_density)

