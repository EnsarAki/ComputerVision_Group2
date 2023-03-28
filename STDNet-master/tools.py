# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import scipy.io as scio
import math
import cv2
import matplotlib.image as mpimg
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt



def compute_cost(y_hat , y):
    
    a = (y_hat - y)     
    
    return 0.5*tf.reduce_mean(tf.reduce_sum( tf.multiply(a , a), axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)



def compute_MAE_and_MSE(y_hat , y_gt):
    
    y_sum = tf.reduce_sum(y_hat , axis=[2,3,4])
    
    y = tf.reduce_sum(y_gt , axis = [2,3,4])
    
    difference = ( y_sum - y )
    
    MAE = tf.reduce_sum(tf.abs(difference))
    
    MSE = tf.reduce_sum(tf.square(difference))
    
    return (MAE , MSE)




def PRL(y_hat , y , ver = None):

    gaussian_w_3 = np.zeros([3,3])
    gaussian_w_3[1,1] = 1
    gaussian_w_3 = cv2.GaussianBlur(gaussian_w_3 , (3 , 3) , 1)
    gaussian_w_3 = np.reshape(gaussian_w_3 , [3,3,1,1])

    gaussian_w_5 = np.zeros([5,5])
    gaussian_w_5[2,2] = 1
    gaussian_w_5 = cv2.GaussianBlur(gaussian_w_5 , (5 , 5) , 1)
    gaussian_w_5 = np.reshape(gaussian_w_5 , [5,5,1,1])

    
    gaussian_w_3 = tf.constant(gaussian_w_3 , tf.float32)
    gaussian_w_5 = tf.constant(gaussian_w_5 , tf.float32)
    
    
    shape = tf.shape(y_hat)
    
    a = (y_hat - y)
    if ver == "L2":
        cost_1 = 0.5*tf.reduce_mean(tf.reduce_sum( tf.multiply(a , a), axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)  
    elif ver == "L1":
        cost_1 = tf.reduce_mean(tf.reduce_sum( tf.abs(a), axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)  

        
    y_hat = tf.reshape(y_hat , [shape[0]* shape[1] , shape[2] , shape[3] , shape[4]])
    
    y = tf.reshape(y , [shape[0]* shape[1] , shape[2] , shape[3] , shape[4]])

    y_hat_gau_3 = tf.compat.v1.nn.conv2d(y_hat , gaussian_w_3 , strides=[1,1,1,1] ,  padding="SAME")
    
    y_gau_3 =  tf.compat.v1.nn.conv2d(y , gaussian_w_3 , strides=[1,1,1,1] ,  padding="SAME")
    
    y_hat_gau_5 = tf.compat.v1.nn.conv2d(y_hat , gaussian_w_5 , strides=[1,1,1,1] ,  padding="SAME")
    
    y_gau_5 =  tf.compat.v1.nn.conv2d(y , gaussian_w_5 , strides=[1,1,1,1] ,  padding="SAME")

    
    y_hat = tf.reshape(y_hat_gau_3 , [shape[0] ,  shape[1] , shape[2] , shape[3] , shape[4]])
    y     = tf.reshape(y_gau_3     , [shape[0] ,  shape[1] , shape[2] , shape[3] , shape[4]])       
    a = (y_hat - y)
    if ver =="L2":
        cost_2 = 0.5*tf.reduce_mean(tf.reduce_sum( tf.multiply(a , a), axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)
    elif ver == "L1":
        cost_2 = tf.reduce_mean(tf.reduce_sum( tf.abs(a) , axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)
 

    y_hat = tf.reshape(y_hat_gau_5 , [shape[0] ,  shape[1] , shape[2] , shape[3] , shape[4]])
    y     = tf.reshape(y_gau_5     , [shape[0] ,  shape[1] , shape[2] , shape[3] , shape[4]])       
    a = (y_hat - y)
    if ver == "L2":
        cost_3 = 0.5*tf.reduce_mean(tf.reduce_sum( tf.multiply(a , a), axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)    
    elif ver == "L1":
        cost_3 = tf.reduce_mean(tf.reduce_sum( tf.abs(a), axis = [2,3,4] ,keepdims=False) , axis = [0,1] , keepdims = False)    
   
        
    return cost_1 + 15 * cost_2 + 3 * cost_3






def random_mini_batches(X,Y,mini_batch_size = 1 , seed = 0):

    np.random.seed(seed)
    n = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(n))
    shuffled_X = X[permutation , : , : , : , :]
    shuffled_Y = Y[permutation , : , : , : , :]
 
    num_full_minibatches = math.floor(n/mini_batch_size)

    for i in range(num_full_minibatches):

        X_mini_batch = shuffled_X[i * mini_batch_size : (i+1)*mini_batch_size , : , : , : , :]
        Y_mini_batch = shuffled_Y[i * mini_batch_size : (i+1)*mini_batch_size , : , : , : , :]
      
        mini_batch = (X_mini_batch , Y_mini_batch)
        mini_batches.append(mini_batch)
      
    return mini_batches


def loadDensityMap(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")
        return

    # Formatting video number and frame number for loading
    vidNum = "{0:0=3d}".format(vidNum - 1)
    frameNum = "{0:0=3d}".format(frameNum)
    loadedDensityMap = np.array(Image.open(f'C:/Users/nik97/Desktop/Computer Vision/vidf-cvpr-density-map/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.png').convert("L"))/255
    return np.reshape(loadedDensityMap, (loadedDensityMap.shape[0], loadedDensityMap.shape[1], 1))

def loadImage(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")
        return

    # Formatting video number and frame number for loading
    vidNum = "{0:0=3d}".format(vidNum - 1)
    frameNum = "{0:0=3d}".format(frameNum)
    loadedImage = mpimg.imread(f'C:/Users/nik97/Desktop/Computer Vision/ucsdpeds/vidf/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.png')
    return np.reshape(loadedImage, (loadedImage.shape[0], loadedImage.shape[1], 1))


def loadROI():
    # Loading video ROI
    loadedROI = loadmat(f'C:/Users/nik97/Desktop/Computer Vision/vidf-cvpr/vidf1_33_roi_mainwalkway.mat')
    loadedROI = loadedROI['roi'][0][0][2]
    return np.reshape(loadedROI, (loadedROI.shape[0], loadedROI.shape[1], 1))

def DataLoader(#x_file_path = "./UCSD_x_data.mat" ,
               #y_file_path = "./UCSD_label_data.mat" ,
               data_aug = False , time_step = 10):
    
    
    X_data_orig = np.zeros((10, 200, 158, 238, 1))
    
    Y_data_orig = np.zeros((10, 200, 158, 238, 1))

    for vidNum in range(1, 11):
        for frameNum in range(1, 201):
            X_data_orig[vidNum - 1][frameNum - 1] = loadImage(vidNum, frameNum)
            Y_data_orig[vidNum - 1][frameNum - 1] = loadDensityMap(vidNum, frameNum)

    ROI = loadROI()

    X = X_data_orig[:, :] * ROI
    
    Y = Y_data_orig[:, :] * ROI


    train_index = [3,4,5,6]
    
    test_index = [0,1,2,7,8,9]


    
    X_train = X[train_index]
    
    Y_train = Y[train_index] 
    
    X_test = X[test_index]
    
    Y_test = Y[test_index] 


    if data_aug == True:
        X_train , Y_train = data_augmentation(X_train,Y_train)
    else:
        pass

    X_train = X_train.reshape([-1, time_step, 158, 238, 1])
    
    X_test = X_test.reshape([-1, time_step, 158, 238, 1])

    Y_train = Y_train.reshape([-1, time_step, 158, 238, 1])
    
    Y_test = Y_test.reshape([-1, time_step, 158, 238, 1])


    
 
    return X_train , Y_train , X_test , Y_test




def data_augmentation(X , Y):
  
    X_flip = np.flip(X , 3)
    Y_flip = np.flip(Y , 3)
    
    X = np.concatenate((X,X_flip),axis = 0)
    Y = np.concatenate((Y,Y_flip),axis = 0)
    
    return X , Y


