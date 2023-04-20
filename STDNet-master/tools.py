import tensorflow as tf
import numpy as np
import scipy.io as scio
import math
import cv2
import matplotlib.image as mpimg
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt


def compute_cost(y_hat, y):
    a = (y_hat - y)

    return 0.5 * tf.compat.v1.reduce_mean(
        tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, a), axis=[2, 3, 4], keepdims=False), axis=[0, 1],
        keepdims=False)


def compute_MAE_and_MSE(y_hat, y_gt):
    y_sum = tf.compat.v1.reduce_sum(y_hat, axis=[2, 3, 4])

    y = tf.compat.v1.reduce_sum(y_gt, axis=[2, 3, 4])

    difference = (y_sum - y)

    MAE = tf.compat.v1.reduce_sum(tf.compat.v1.abs(difference))

    MSE = tf.compat.v1.reduce_sum(tf.compat.v1.square(difference))

    return (MAE, MSE)


def PRL(y_hat, y, ver=None):
    gaussian_w_3 = np.zeros([3, 3])
    gaussian_w_3[1, 1] = 1
    gaussian_w_3 = cv2.GaussianBlur(gaussian_w_3, (3, 3), 1)
    gaussian_w_3 = np.reshape(gaussian_w_3, [3, 3, 1, 1])

    gaussian_w_5 = np.zeros([5, 5])
    gaussian_w_5[2, 2] = 1
    gaussian_w_5 = cv2.GaussianBlur(gaussian_w_5, (5, 5), 1)
    gaussian_w_5 = np.reshape(gaussian_w_5, [5, 5, 1, 1])

    gaussian_w_3 = tf.compat.v1.constant(gaussian_w_3, tf.float32)
    gaussian_w_5 = tf.compat.v1.constant(gaussian_w_5, tf.float32)

    shape = tf.compat.v1.shape(y_hat)

    a = (y_hat - y)
    if ver == "L2":
        cost_1 = 0.5 * tf.compat.v1.reduce_mean(
            tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, a), axis=[2, 3, 4], keepdims=False), axis=[0, 1],
            keepdims=False)
    elif ver == "L1":
        cost_1 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.abs(a), axis=[2, 3, 4], keepdims=False),
                                          axis=[0, 1], keepdims=False)

    y_hat = tf.compat.v1.reshape(y_hat, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

    y = tf.compat.v1.reshape(y, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

    y_hat_gau_3 = tf.compat.v1.nn.conv2d(y_hat, gaussian_w_3, strides=[1, 1, 1, 1], padding="SAME")

    y_gau_3 = tf.compat.v1.nn.conv2d(y, gaussian_w_3, strides=[1, 1, 1, 1], padding="SAME")

    y_hat_gau_5 = tf.compat.v1.nn.conv2d(y_hat, gaussian_w_5, strides=[1, 1, 1, 1], padding="SAME")

    y_gau_5 = tf.compat.v1.nn.conv2d(y, gaussian_w_5, strides=[1, 1, 1, 1], padding="SAME")

    y_hat = tf.compat.v1.reshape(y_hat_gau_3, [shape[0], shape[1], shape[2], shape[3], shape[4]])
    y = tf.compat.v1.reshape(y_gau_3, [shape[0], shape[1], shape[2], shape[3], shape[4]])
    a = (y_hat - y)
    if ver == "L2":
        cost_2 = 0.5 * tf.compat.v1.reduce_mean(
            tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, a), axis=[2, 3, 4], keepdims=False), axis=[0, 1],
            keepdims=False)
    elif ver == "L1":
        cost_2 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.abs(a), axis=[2, 3, 4], keepdims=False),
                                          axis=[0, 1], keepdims=False)

    y_hat = tf.compat.v1.reshape(y_hat_gau_5, [shape[0], shape[1], shape[2], shape[3], shape[4]])
    y = tf.compat.v1.reshape(y_gau_5, [shape[0], shape[1], shape[2], shape[3], shape[4]])
    a = (y_hat - y)
    if ver == "L2":
        cost_3 = 0.5 * tf.compat.v1.reduce_mean(
            tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, a), axis=[2, 3, 4], keepdims=False), axis=[0, 1],
            keepdims=False)
    elif ver == "L1":
        cost_3 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.abs(a), axis=[2, 3, 4], keepdims=False),
                                          axis=[0, 1], keepdims=False)

    return cost_1 + 15 * cost_2 + 3 * cost_3


def random_mini_batches(X, Y, mini_batch_size=1, seed=0):
    np.random.seed(seed)
    n = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(n))
    shuffled_X = X[permutation, :, :, :, :]
    shuffled_Y = Y[permutation, :, :, :, :]

    num_full_minibatches = math.floor(n / mini_batch_size)

    for i in range(num_full_minibatches):
        X_mini_batch = shuffled_X[i * mini_batch_size: (i + 1) * mini_batch_size, :, :, :, :]
        Y_mini_batch = shuffled_Y[i * mini_batch_size: (i + 1) * mini_batch_size, :, :, :, :]

        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)

    return mini_batches


def loadImage(dataset, vidNum=None, frameNum=None):
    if dataset == 'ucsd':
        # Checking if video and frame number is lower than 1
        if frameNum < 1 or vidNum < 1:
            print("Video or frame number smaller than 1 does not exist.")
            return

        # Formatting video number and frame number for loading
        vidNum = "{0:0=3d}".format(vidNum - 1)
        frameNum = "{0:0=3d}".format(frameNum)
        loadedImage = mpimg.imread(f'ucsdpeds/vidf/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.png')
        return np.reshape(loadedImage, (loadedImage.shape[0], loadedImage.shape[1], 1))
    elif dataset == 'mall':
        if frameNum < 1:
            print("Video or frame number smaller than 1 does not exist.")
            return
        frameNum = "{0:0=6d}".format(frameNum)
        loadedImage = np.array(Image.open(f'mall_dataset/frames/seq_{frameNum}.jpg').convert("L")) / 255
        return np.reshape(loadedImage, (loadedImage.shape[0], loadedImage.shape[1], 1))
    else:
        raise SystemExit('Cannot load image without dataset.')


def loadDensityMap(dataset, vidNum=None, frameNum=None):
    if dataset == 'ucsd':
        # Checking if video and frame number is lower than 1
        if frameNum < 1 or vidNum < 1:
            print("Video or frame number smaller than 1 does not exist.")
            return
        # Formatting video number and frame number for loading
        vidNum = "{0:0=3d}".format(vidNum - 1)
        frameNum = "{0:0=3d}".format(frameNum)
        loadedDensityMap = np.load(f'vidf-cvpr-density-map/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.npy')
        return np.reshape(loadedDensityMap, (loadedDensityMap.shape[0], loadedDensityMap.shape[1], 1))
    elif dataset == 'mall':
        if frameNum < 1:
            print("Video or frame number smaller than 1 does not exist.")
            return
        frameNum = "{0:0=6d}".format(frameNum)
        loadedDensityMap = np.load(f'mall_dataset-density-map/seq_{frameNum}.npy')
        return np.reshape(loadedDensityMap, (loadedDensityMap.shape[0], loadedDensityMap.shape[1], 1))
    else:
        raise SystemExit('Cannot load density map without dataset.')


def loadROI(dataset):
    # Loading video ROI
    if dataset == 'ucsd':
        loadedROI = loadmat(f'vidf-cvpr/vidf1_33_roi_mainwalkway.mat')
        loadedROI = loadedROI['roi'][0][0][2]
    elif dataset == 'mall':
        loadedROI = loadmat(f'mall_dataset/perspective_roi.mat')
        loadedROI = (loadedROI['roi'][0][0])[0]
    return np.reshape(loadedROI, (loadedROI.shape[0], loadedROI.shape[1], 1))


def DataLoader(dataset, data_aug=False, time_step=10):

    if dataset == 'ucsd':
        frame_size = (158, 238)
        X_data_orig = np.zeros((10, 200, frame_size[0], frame_size[1], 1))
        Y_data_orig = np.zeros((10, 200, frame_size[0], frame_size[1], 1))
        for vidNum in range(1, 11):
            for frameNum in range(1, 201):
                X_data_orig[vidNum - 1][frameNum - 1] = loadImage(dataset, vidNum=vidNum, frameNum=frameNum)
                Y_data_orig[vidNum - 1][frameNum - 1] = loadDensityMap(dataset, vidNum=vidNum, frameNum=frameNum)

    elif dataset == 'mall':
        frame_size = (480, 640)
        X_data_orig = np.zeros((1, 2000, frame_size[0], frame_size[1], 1))
        Y_data_orig = np.zeros((1, 2000, frame_size[0], frame_size[1], 1))
        for frameNum in range(1, 2001):
            X_data_orig[0][frameNum - 1] = loadImage(dataset, frameNum=frameNum)
            Y_data_orig[0][frameNum - 1] = loadDensityMap(dataset, frameNum=frameNum)
        X_data_orig = X_data_orig.reshape(10, 200, frame_size[0], frame_size[1], 1)
        Y_data_orig = Y_data_orig.reshape(10, 200, frame_size[0], frame_size[1], 1)

    else:
        raise SystemExit('A valid dataset was not chosen.')

    ROI = loadROI(dataset)

    X = X_data_orig[:, :] * ROI

    Y = Y_data_orig[:, :] * ROI

    train_index = [3, 4, 5, 6]

    test_index = [0, 1, 2, 7, 8, 9]

    X_train = X[train_index]

    Y_train = Y[train_index]

    X_test = X[test_index]

    Y_test = Y[test_index]

    if data_aug == True:
        X_train, Y_train = data_augmentation(X_train, Y_train)
    else:
        pass

    X_train = X_train.reshape([-1, time_step, frame_size[0], frame_size[1], 1]).astype(np.float32)

    X_test = X_test.reshape([-1, time_step, frame_size[0], frame_size[1], 1]).astype(np.float32)

    Y_train = Y_train.reshape([-1, time_step, frame_size[0], frame_size[1], 1]).astype(np.float32)

    Y_test = Y_test.reshape([-1, time_step, frame_size[0], frame_size[1], 1]).astype(np.float32)


    return X_train, Y_train, X_test, Y_test, frame_size


def data_augmentation(X, Y):
    X_flip = np.flip(X, 3)
    Y_flip = np.flip(Y, 3)

    X = np.concatenate((X, X_flip), axis=0)
    Y = np.concatenate((Y, Y_flip), axis=0)

    return X, Y