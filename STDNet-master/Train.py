# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as scio
import scipy
from PIL import Image
import time
import matplotlib.pyplot as plt
from matplotlib import cm as CM

from VGG_backbone import VGG_10
import tools


def resize_and_adjust_channel(x, resize_shape=None, channel=None, Name=None):
    with tf.compat.v1.variable_scope("Adjust_Channel_" + Name):
        x_resize = tf.compat.v1.image.resize_bilinear(x, resize_shape)

        z = tf.compat.v1.layers.conv2d(x_resize, channel, [1, 1], strides=[1, 1], padding="SAME")

        return z


def output_adjust(x, resize=None, channel=None, Name=None):
    with tf.compat.v1.variable_scope("Adjust_Channel_" + Name):
        z = tf.compat.v1.layers.conv2d(x, channel, [1, 1], strides=[1, 1], padding="SAME")

        z_resize = tf.compat.v1.image.resize_bilinear(z, resize)

        return z_resize


def common_conv2d(z, in_filter=None, out_filter=None, Name=None):
    with tf.compat.v1.variable_scope(Name):
        W = tf.compat.v1.get_variable(name=Name + "_W", shape=[1, 1, in_filter, out_filter])
        b = tf.compat.v1.get_variable(name=Name + "_b", shape=[out_filter],
                                      initializer=tf.compat.v1.zeros_initializer())

        z = tf.compat.v1.nn.conv2d(z, W, strides=[1, 1, 1, 1], padding="SAME") + b
        z = tf.compat.v1.nn.relu(z)

        return z


def dilated_conv2d(z, in_filter=None, out_filter=None, dilated_rate=None, Name=None):
    with tf.compat.v1.variable_scope(Name):
        W = tf.compat.v1.get_variable(name=Name + "_W", shape=[3, 3, in_filter, out_filter])
        b = tf.compat.v1.get_variable(name=Name + "_b", shape=[out_filter],
                                      initializer=tf.compat.v1.zeros_initializer())

        z = tf.compat.v1.nn.atrous_conv2d(z, W, rate=dilated_rate, padding="SAME") + b
        z = tf.compat.v1.nn.relu(z)

        return z


def common_conv3d(z, in_filter=None, out_filter=None, Name=None):
    with tf.compat.v1.variable_scope(Name):
        W = tf.compat.v1.get_variable(name=Name + "_W", shape=[1, 1, 1, in_filter, out_filter])
        #        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())

        z = tf.compat.v1.nn.conv3d(z, W, strides=[1, 1, 1, 1, 1], padding="SAME")
        z = tf.compat.v1.nn.relu(z)
        return z


def dilated_conv3d(z, in_filter=None, out_filter=None, dilated_rate=None, Name=None):
    with tf.compat.v1.variable_scope(Name):

        zero = tf.compat.v1.zeros([1, 1, 1, in_filter, out_filter])
        W1 = tf.compat.v1.get_variable(name=Name + "_W1", shape=[1, 1, 1, in_filter, out_filter])
        W2 = tf.compat.v1.get_variable(name=Name + "_W2", shape=[1, 1, 1, in_filter, out_filter])
        W3 = tf.compat.v1.get_variable(name=Name + "_W3", shape=[1, 1, 1, in_filter, out_filter])
        #        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())

        if dilated_rate == 1:
            W = tf.compat.v1.concat([W1, W2, W3], axis=0)
        elif dilated_rate == 2:
            W = tf.compat.v1.concat([W1, zero, W2, zero, W3], axis=0)
        elif dilated_rate == 3:
            W = tf.compat.v1.concat([W1, zero, zero, W2, zero, zero, W3], axis=0)

        z = tf.compat.v1.nn.conv3d(z, W, strides=[1, 1, 1, 1, 1], padding="SAME")
        z = tf.compat.v1.nn.relu(z)
        return z


def Dense_Spatial_Block(x, name=None):
    channel_list = [64, 64, 64]
    with tf.compat.v1.variable_scope("Dense_Spatial_Block" + "_" + name):
        z1 = common_conv2d(x, 512, 256, "DSB_1-1")
        z1 = dilated_conv2d(z1, 256, channel_list[0], 1, "DSB_1-2")

        z2 = tf.compat.v1.concat([x, z1], axis=3)
        z2 = common_conv2d(z2, 512 + channel_list[0], 256, "DSB_2-1")
        z2 = dilated_conv2d(z2, 256, channel_list[1], 2, "DSB_2-2")

        z3 = tf.compat.v1.concat([x, z1, z2], axis=3)
        z3 = common_conv2d(z3, 512 + channel_list[0] + channel_list[1], 256, "DSB_3-1")
        z3 = dilated_conv2d(z3, 256, channel_list[2], 3, "DSB_3-2")

        z4 = tf.compat.v1.concat([x, z1, z2, z3], axis=3)
        z4 = common_conv2d(z4, 512 + channel_list[0] + channel_list[1] + channel_list[2], 512, "DSB_4-1")

        return z4


def Spatial_Channel_Aware_Block(x, name=None):
    with tf.compat.v1.variable_scope("Spatial_Channel_Aware_Block" + "_" + name):
        gap = tf.compat.v1.reduce_mean(x, axis=[1, 2], keep_dims=True)
        gap = tf.compat.v1.reshape(gap, [tf.compat.v1.shape(x)[0], tf.compat.v1.shape(x)[3]])

        weight = tf.compat.v1.layers.dense(gap, 128)
        weight = tf.compat.v1.nn.relu(weight)
        weight = tf.compat.v1.layers.dense(weight, 512)
        weight = tf.compat.v1.nn.sigmoid(weight)

        a = tf.compat.v1.reshape(weight, [tf.compat.v1.shape(x)[0], 1, 1, tf.compat.v1.shape(x)[3]])

        z = tf.compat.v1.multiply(a, x)

        return z


def Dense_Temporal_Block(x, name=None):
    channel_list = [64, 64, 64]
    with tf.compat.v1.variable_scope("Dense_Temporal_Block" + "_" + name):
        z1 = common_conv3d(x, 512, 256, "DTB_1-1")
        z1 = dilated_conv3d(z1, 256, channel_list[0], 1, "DTB_1-2")

        z2 = tf.compat.v1.concat([x, z1], axis=4)
        z2 = common_conv3d(z2, 512 + channel_list[0], 256, "DTB_2-1")
        z2 = dilated_conv3d(z2, 256, channel_list[1], 2, "DTB_2-2")

        z3 = tf.compat.v1.concat([x, z1, z2], axis=4)
        z3 = common_conv3d(z3, 512 + channel_list[0] + channel_list[1], 256, "DTB_3-1")
        z3 = dilated_conv3d(z3, 256, channel_list[2], 3, "DTB_3-2")

        z4 = tf.compat.v1.concat([x, z1, z2, z3], axis=4)
        z4 = common_conv3d(z4, 512 + channel_list[0] + channel_list[1] + channel_list[2], 512, "DTB_4-1")

        return z4


def Temporal_Channel_Aware_Block(x, name=None):
    with tf.compat.v1.variable_scope("Temporal_Channel_Aware_Block" + "_" + name):
        gap = tf.compat.v1.reduce_mean(x, axis=[1, 2, 3], keep_dims=True)
        gap = tf.compat.v1.reshape(gap, [tf.compat.v1.shape(x)[0], tf.compat.v1.shape(x)[4]])

        weight = tf.compat.v1.layers.dense(gap, 128)
        weight = tf.compat.v1.nn.relu(weight)
        weight = tf.compat.v1.layers.dense(weight, 512)
        weight = tf.compat.v1.nn.sigmoid(weight)

        a = tf.compat.v1.reshape(weight, [tf.compat.v1.shape(x)[0], 1, 1, 1, tf.compat.v1.shape(x)[4]])

        z = tf.compat.v1.multiply(a, x)

        return z


def Training(dataset, batch_SIZE=None, time_step=None, Epoch=None, lr=None):
    X_train, Y_train, X_test, Y_test, frame_size = tools.DataLoader(data_aug=True, time_step=time_step, dataset=dataset)

    VGG = VGG_10()

    x = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[batch_SIZE, time_step, frame_size[0], frame_size[1], 1])
    y = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[batch_SIZE, time_step, frame_size[0], frame_size[1], 1])

    x_reshape = tf.compat.v1.reshape(x, [-1, frame_size[0], frame_size[1], 1])

    LR = tf.compat.v1.placeholder(tf.compat.v1.float32)

    z = resize_and_adjust_channel(x_reshape, [316, 476], 3, "Start")
    z = VGG.forward(z)

    S_1 = Dense_Spatial_Block(z, "DSB_1")
    S_1 = Spatial_Channel_Aware_Block(S_1, "SCA_1")
    z_1 = S_1

    z = tf.compat.v1.reshape(z_1, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_1 = Dense_Temporal_Block(z, "DTB_1")
    T_1 = Temporal_Channel_Aware_Block(T_1, "TCA_1")
    z_1 = T_1

    z = tf.compat.v1.reshape(z_1, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    S_2 = Dense_Spatial_Block(z, "DSB_2")
    S_2 = Spatial_Channel_Aware_Block(S_2, "SCA_2")
    z_2 = S_2

    z = tf.compat.v1.reshape(z_2, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_2 = Dense_Temporal_Block(z, "DTB_2")
    T_2 = Temporal_Channel_Aware_Block(T_2, "TCA_2")
    z_2 = T_2

    z = tf.compat.v1.reshape(z_2, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    S_3 = Dense_Spatial_Block(z, "DSB_3")
    S_3 = Spatial_Channel_Aware_Block(S_3, "SCA_3")
    z_3 = S_3

    z = tf.compat.v1.reshape(z_3, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_3 = Dense_Temporal_Block(z, "DTB_3")
    T_3 = Temporal_Channel_Aware_Block(T_3, "TCA_3")
    z_3 = T_3

    z = tf.compat.v1.reshape(z_3, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    S_4 = Dense_Spatial_Block(z, "DSB_4")
    S_4 = Spatial_Channel_Aware_Block(S_4, "SCA_4")
    z_4 = S_4

    z = tf.compat.v1.reshape(z_4, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_4 = Dense_Temporal_Block(z, "DTB_4")
    T_4 = Temporal_Channel_Aware_Block(T_4, "TCA_4")
    z_4 = T_4

    z = tf.compat.v1.reshape(z_4, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    z = dilated_conv2d(z, 512, 128, 1, "128")
    z = dilated_conv2d(z, 128, 64, 1, "64")
    z = output_adjust(z, [frame_size[0], frame_size[1]], 1, "End")
    z = tf.compat.v1.reshape(z, [batch_SIZE, time_step, frame_size[0], frame_size[1], 1])

    all_variable = tf.compat.v1.trainable_variables()
    Reg_Loss = 1e-4 * tf.compat.v1.reduce_sum([tf.compat.v1.nn.l2_loss(v) for v in all_variable])

    cost = tools.PRL(z, y, "L1")
    cost = cost + Reg_Loss

    performance = tools.compute_MAE_and_MSE(z, y)

    predicted_images = z
    validation_images = y

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(cost)

    initial = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:

        best_cost = 0

        print("-----------------------------------------------------------------------------\n")
        print("\nStart Training...\n")
        print("Number of parameters : ",
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]), "\n")

        Time = 0
        seed = 0

        saver = tf.compat.v1.train.Saver()

        # train_writer = tf.compat.v1.summary.FileWriter("./logs/train", sess.graph)
        # test_writer = tf.compat.v1.summary.FileWriter("./logs/test", sess.graph)

        sess.graph.finalize()
        sess.run(initial)

        for epoch in range(Epoch + 1):
            if epoch == 30:
                lr = lr / 2
            if epoch == 60:
                lr = lr / 2
            if epoch == 100:
                lr = lr / 2

            start_time = time.time()

            mini_batch_cost = 0
            mini_batch_MAE = 0
            mini_batch_MSE = 0

            seed = seed + 1

            minibatches = tools.random_mini_batches(X_train, Y_train, batch_SIZE, seed=seed)

            for data in minibatches:
                (X_train_batch, Y_train_batch) = data

                _, temp_cost, train_performance, train_prediction, train_validation = sess.run(
                    [optimizer, cost, performance, predicted_images, validation_images],
                    feed_dict={x: X_train_batch,
                               y: Y_train_batch,
                               LR: lr,
                               })
                # fig = plt.figure()
                # fig.add_subplot(1, 2, 1)
                # plt.imshow(train_prediction[0, 0], cmap='gray')
                # plt.title("Prediction")
                #
                # fig.add_subplot(1, 2, 2)
                # plt.imshow(train_validation[0, 0], cmap='gray')
                # plt.title("Validation")
                #
                # plt.show()

                # print(f'Prediction sum: {np.sum(train_prediction[0, 0])} Validation sum: {np.sum(train_validation[0, 0])}')

                mini_batch_cost += temp_cost * batch_SIZE * time_step / (X_train.shape[0] * X_train.shape[1])
                mini_batch_MAE += train_performance[0] / (X_train.shape[0] * X_train.shape[1])
                mini_batch_MSE += train_performance[1] / (X_train.shape[0] * X_train.shape[1])

            total_cost = round(mini_batch_cost, 7)
            total_MAE = round(mini_batch_MAE, 4)
            total_MSE = round(np.sqrt(mini_batch_MSE), 4)

            print("Epoch : ", epoch, " , Cost :  ", total_cost, " , MAE : ", total_MAE, ", MSE : ", total_MSE)
            if best_cost == 0:
                best_cost = total_cost
                saver.save(sess, './Model/model.ckpt')
                print("Checkpoint saved")
            elif best_cost > total_cost:
                best_cost = total_cost
                saver.save(sess, './Model/model.ckpt')
                print("Checkpoint saved")

            if True:

                test_cost, test_MAE, test_MSE = 0, 0, 0
                test_batches = tools.random_mini_batches(X_test, Y_test, batch_SIZE, seed=seed)

                for i in test_batches:
                    (X_test_batch, Y_test_batch) = i

                    temp_cost, test_performance = sess.run([cost, performance], feed_dict={x: X_test_batch,
                                                                                           y: Y_test_batch,
                                                                                           })

                    test_cost += temp_cost * batch_SIZE * time_step / (X_test.shape[0] * X_test.shape[1])
                    test_MAE += test_performance[0] / (X_test.shape[0] * X_test.shape[1])
                    test_MSE += test_performance[1] / (X_test.shape[0] * X_test.shape[1])

                test_cost = round(test_cost, 7)
                test_MAE = round(test_MAE, 4)
                test_MSE = round(np.sqrt(test_MSE), 4)

                print("Testing , cost :  ", test_cost, " , MAE : ", test_MAE, " , MSE : ", test_MSE, "\n")

            process_time = time.time() - start_time
            Time = Time + (process_time - Time) / (epoch + 1)

            if epoch % 5 == 0:
                print("Average training time  per epoch : ", Time)
    print("Done.\n")


def load_pretrained_model(dataset, batch_SIZE=None, time_step=None, lr=None, save_predictions=False):
    X_train, Y_train, X_test, Y_test, frame_size = tools.DataLoader(data_aug=True, time_step=time_step, dataset=dataset)

    VGG = VGG_10()

    x = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[batch_SIZE, time_step, frame_size[0], frame_size[1], 1])
    y = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[batch_SIZE, time_step, frame_size[0], frame_size[1], 1])

    x_reshape = tf.compat.v1.reshape(x, [-1, frame_size[0], frame_size[1], 1])

    LR = tf.compat.v1.placeholder(tf.compat.v1.float32)

    z = resize_and_adjust_channel(x_reshape, [316, 476], 3, "Start")
    z = VGG.forward(z)

    S_1 = Dense_Spatial_Block(z, "DSB_1")
    S_1 = Spatial_Channel_Aware_Block(S_1, "SCA_1")
    z_1 = S_1

    z = tf.compat.v1.reshape(z_1, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_1 = Dense_Temporal_Block(z, "DTB_1")
    T_1 = Temporal_Channel_Aware_Block(T_1, "TCA_1")
    z_1 = T_1

    z = tf.compat.v1.reshape(z_1, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    S_2 = Dense_Spatial_Block(z, "DSB_2")
    S_2 = Spatial_Channel_Aware_Block(S_2, "SCA_2")
    z_2 = S_2

    z = tf.compat.v1.reshape(z_2, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_2 = Dense_Temporal_Block(z, "DTB_2")
    T_2 = Temporal_Channel_Aware_Block(T_2, "TCA_2")
    z_2 = T_2

    z = tf.compat.v1.reshape(z_2, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    S_3 = Dense_Spatial_Block(z, "DSB_3")
    S_3 = Spatial_Channel_Aware_Block(S_3, "SCA_3")
    z_3 = S_3

    z = tf.compat.v1.reshape(z_3, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_3 = Dense_Temporal_Block(z, "DTB_3")
    T_3 = Temporal_Channel_Aware_Block(T_3, "TCA_3")
    z_3 = T_3

    z = tf.compat.v1.reshape(z_3, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    S_4 = Dense_Spatial_Block(z, "DSB_4")
    S_4 = Spatial_Channel_Aware_Block(S_4, "SCA_4")
    z_4 = S_4

    z = tf.compat.v1.reshape(z_4, [batch_SIZE, time_step, tf.compat.v1.shape(z)[1], tf.compat.v1.shape(z)[2],
                                   tf.compat.v1.shape(z)[3]], name="Reshape_S_T")

    T_4 = Dense_Temporal_Block(z, "DTB_4")
    T_4 = Temporal_Channel_Aware_Block(T_4, "TCA_4")
    z_4 = T_4

    z = tf.compat.v1.reshape(z_4, [-1, tf.compat.v1.shape(z)[2], tf.compat.v1.shape(z)[3], tf.compat.v1.shape(z)[4]])

    z = dilated_conv2d(z, 512, 128, 1, "128")
    z = dilated_conv2d(z, 128, 64, 1, "64")
    z = output_adjust(z, [frame_size[0], frame_size[1]], 1, "End")
    z = tf.compat.v1.reshape(z, [batch_SIZE, time_step, frame_size[0], frame_size[1], 1])

    all_variable = tf.compat.v1.trainable_variables()
    Reg_Loss = 1e-4 * tf.compat.v1.reduce_sum([tf.compat.v1.nn.l2_loss(v) for v in all_variable])

    cost = tools.PRL(z, y, "L1")
    cost = cost + Reg_Loss

    performance = tools.compute_MAE_and_MSE(z, y)

    predicted_images = z
    validation_images = y

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(cost)

    with tf.compat.v1.Session() as sess:

        Time = 0
        seed = 0
        counter = 0

        saver = tf.compat.v1.train.import_meta_graph('./Model/model.ckpt.meta')
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./Model'))

        start_time = time.time()

        if True:
            test_cost, test_MAE, test_MSE = 0, 0, 0
            test_batches = tools.random_mini_batches(X_test, Y_test, batch_SIZE, seed=seed)

            for i in test_batches:
                (X_test_batch, Y_test_batch) = i

                temp_cost, test_performance, test_prediction, test_validation = sess.run(
                    [cost, performance, predicted_images, validation_images],
                    feed_dict={x: X_test_batch, y: Y_test_batch, })

                if save_predictions:
                    for index in range(10):
                        np.save(f'./Prediction/{counter}', test_prediction[0][index])
                        np.save(f'./Validation/{counter}', test_validation[0][index])
                        counter += 1

                # print(f'Prediction sum: {np.sum(test_prediction[0, 0])} Validation sum: {np.sum(test_validation[0, 0])}')

                test_cost += temp_cost * batch_SIZE * time_step / (X_test.shape[0] * X_test.shape[1])
                test_MAE += test_performance[0] / (X_test.shape[0] * X_test.shape[1])
                test_MSE += test_performance[1] / (X_test.shape[0] * X_test.shape[1])

            test_cost = round(test_cost, 7)
            test_MAE = round(test_MAE, 4)
            test_MSE = round(np.sqrt(test_MSE), 4)

            print("Testing , cost :  ", test_cost, " , MAE : ", test_MAE, " , MSE : ", test_MSE, "\n")

        # saver.save(sess, './Model/model.ckpt')
    print("Done.\n")


def createPlots():
    fig = plt.figure()
    for i in range(1, 1201):
        prediction = np.load(f'./Prediction/{i}.npy')
        validation = np.load(f'./Validation/{i}.npy')

        fig.add_subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(prediction, cmap=CM.jet)
        plt.title("Prediction")

        fig.add_subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(validation, cmap=CM.jet)
        plt.title("Validation")

        plt.savefig(f'./Plots/{i}', bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    batch_SIZE = 1

    time_step = 10  # 10 for UCSD, 8 for mall

    Epoch = 2

    lr = 1e-4

    dataset = 'mall'

    # Training(batch_SIZE=batch_SIZE,
    # time_step=time_step,
    # Epoch=Epoch,
    # lr=lr,
    # dataset=dataset)

    load_pretrained_model(batch_SIZE=batch_SIZE,
                          time_step=time_step,
                          lr=lr,
                          dataset=dataset,
                          save_predictions=True)

    # createPlots()
