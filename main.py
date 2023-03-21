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
from PIL import Image
import re
import tensorflow as tf
from tensorflow import keras
from keras import layers

import io
import imageio
from ipywidgets import widgets, Layout, HBox


def loadFrameLoc(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")
        return
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
        return

    # Formatting video number and frame number for loading
    vidNum = "{0:0=3d}".format(vidNum - 1)
    frameNum = "{0:0=3d}".format(frameNum)
    return mpimg.imread(f'ucsdpeds/vidf/vidf1_33_{vidNum}.y/vidf1_33_{vidNum}_f{frameNum}.png')

def loadDensityMap(vidNum, frameNum):
    # Checking if video and frame number is lower than 1
    if frameNum < 1 or vidNum < 1:
        print("Video or frame number smaller than 1 does not exist.")
        return

    # Formatting video number and frame number for loading
    vidNum = "{0:0=1d}".format(vidNum - 1)
    frameNum = "{0:0=1d}".format(frameNum)
    return np.array(Image.open(f'vidf-cvpr-density-map/vidf1_33_{vidNum}_f{frameNum}.png').convert("L"))/255

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    Dmap = np.flip(loadDmap())

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            # sigma = Dmap[pt[1], pt[0]]*0.3
             sigma = 4.5
            # sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

def prepareImages():
    for vidNum in range(1, 11):
        for frameNum in range(1, 201):
            img_density = np.zeros((158, 238))
            img = loadImage(vidNum, frameNum)
            location = loadFrameLoc(vidNum, frameNum)

            for loc in location:
                if int(loc[0]) < 238 and int(loc[1]) < 158:
                    img_density[int(loc[1]), int(loc[0])] = 1
                # else:
                    # print("Person removed")
            img_density = gaussian_filter_density(img_density)

            # plt.imshow(img_density, cmap=CM.jet)
            # plt.show()
            # print(location.shape[0] - np.sum(img_density))

            mpimg.imsave(f'vidf-cvpr-density-map/vidf1_33_{"{0:0=3d}".format(vidNum - 1)}.y/vidf1_33_{"{0:0=3d}".format(vidNum - 1)}_f{"{0:0=3d}".format(frameNum)}.png', img_density, format='png', cmap="gray")


def create_shifted_frames(data):
    x = data[:, 0:data.shape[1] - 1, :, :]
    y = data[:, 1:data.shape[1], :, :]
    return x, y


if __name__ == '__main__':
    # prepareImages()

    batch_size = 4
    from_frame = int(600/batch_size)
    to_frame = int(1400/batch_size)

    training_dataset = keras.utils.image_dataset_from_directory('ucsdpeds/vidf',
                                                                image_size=(158, 238),
                                                                color_mode="grayscale",
                                                                shuffle=False,
                                                                batch_size=batch_size)

    validation_dataset = keras.utils.image_dataset_from_directory('vidf-cvpr-density-map',
                                                                  image_size=(158, 238),
                                                                  color_mode="grayscale",
                                                                  shuffle=False,
                                                                  batch_size=batch_size)

    train_dataset = np.zeros((200, 4, 158, 238, 1))
    val_dataset = np.zeros((200, 4, 158, 238, 1))
    test_training_dataset = np.zeros((150, 4, 158, 238, 1))
    test_validation_dataset = np.zeros((150, 4, 158, 238, 1))

    count = 0
    test = 0
    for images, labels in training_dataset.take(to_frame):
        count += 1
        if count < (from_frame + 1):
            images = tf.cast(images / 255., tf.float32)
            test_training_dataset[count - (from_frame + 1)] = images
            continue
        # for i in range(batch_size):
        #     print(training_dataset.class_names[labels[i]])
        #     test += 1
        images = tf.cast(images / 255., tf.float32)
        train_dataset[count - (from_frame + 1)] = images
    # print(f'Loaded a total of {test} images and {count - from_frame} batches to the training dataset')

    count = 0
    test = 0
    for images, labels in validation_dataset.take(to_frame):
        count += 1
        if count < (from_frame + 1):
            images = tf.cast(images / 255., tf.float32)
            test_validation_dataset[count - (from_frame + 1)] = images
            continue
        # for i in range(batch_size):
        #     print(validation_dataset.class_names[labels[i]])
        #     test += 1
        images = tf.cast(images / 255., tf.float32)
        val_dataset[count - (from_frame + 1)] = images
    # print(f'Loaded a total of {test} images and {count - from_frame} batches to the validation dataset')

    # x_train, y_train = create_shifted_frames(train_dataset)
    # x_val, y_val = create_shifted_frames(val_dataset)

    print("Training Dataset Shapes: " + str(train_dataset.shape))
    print("Validation Dataset Shapes: " + str(val_dataset.shape))

    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *train_dataset.shape[2:]))

    # We will construct 4 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=128,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)

    x = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), activation="sigmoid", padding="same")(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

    # # Define some callbacks to improve training.
    # early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define modifiable training hyperparameters.
    epochs = 20
    batch_size = 4

    # Fit the model to the training data.
    model.fit(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        #callbacks=[early_stopping, reduce_lr],
    )

    model.save('Model/trained_model')

    model = keras.models.load_model('Model/trained_model')

    # Pick the first/last four frames from the example.
    frames = test_training_dataset[0]
    original_frames = test_validation_dataset[0]

    # Predict a new set of 4 frames.
    for _ in range(4):
        # Extract the model's prediction and post-process it.
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame), axis=0)

    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 4, figsize=(20, 4))

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    # Plot the new frames.
    new_frames = frames[4:, ...]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    # Display the figure.
    plt.show()


