from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from os.path import dirname, join as pjoin
import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm as CM
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

def loadDmap():
    # Loading video Dmap
    mat_fname = pjoin('vidf-cvpr', f'vidf1_33_dmap3.mat')
    loadedDmap = loadmat(mat_fname)
    return loadedDmap['dmap'][0][0][0]

def loadImage(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")

    # Formatting video number and frame number for loading
    vidNum = "{0:0=3d}".format(vidNum - 1)
    frameNum = "{0:0=3d}".format(frameNum)
    return mpimg.imread(f'ucsdpeds/vidf/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.png')

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    Dmap = np.flip(loadDmap())

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            # sigma = Dmap[pt[1], pt[0]]*0.3
            sigma = 4.8
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density


if __name__ == '__main__':
    loc_coordinates = np.zeros(2)

    for vidNum in range(1, 11):
        for frameNum in range(1, 201):
            img_density = np.zeros((158, 238))
            img = loadImage(vidNum, frameNum)
            location = loadFrameLoc(vidNum, frameNum)

            for loc in location:
                if int(loc[0]) < 238 and int(loc[1]) < 158:
                    img_density[int(loc[1]), int(loc[0])] = 1
                else:
                    print("Person removed")
            img_density = gaussian_filter_density(img_density)

            # plt.imshow(img_density, cmap=CM.jet)
            # plt.show()

            mpimg.imsave(f'vidf-cvpr-density-map/vidf1_33_{vidNum}_f{frameNum}.png', img_density, cmap=CM.jet)

