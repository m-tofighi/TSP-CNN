# This code is for testing the trained TSP-CNN network. 
# How to run: 
# 	1- Download the trained network models (checkpoint) for each dataset,
# 	2- Modify the directories to the paths containing the trained model and test data,
# 	3- Specify a path to save the outputs
#	4- For assessment we used the same Matlab code provided by Sirinukunwattana et al.
#
# For further information and questions please contact M. Tofighi at tofighi@psu.edu

import numpy as np
import tensorflow as tf
import glob
import cv2
import ntpath
EPOCH_SIZE = 300

TEST_PATH = './test_data/' # Directory of test data
ALL_MODEL_PATH = './record/18-05-27-20-10/model/' # Directory of trained model
ALL_TEST_SAVE_PATH = './output/18-05-27-20-10/' # Directory to save test results

SHAPE_PATH_TRAIN = './SHAPE.npy'
SHAPE_TRAIN = np.load(SHAPE_PATH_TRAIN)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

with tf.Session() as sess:
	# Weights 
    w_conv1 = tf.get_variable(name="w_conv1", shape=[5, 5, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv2 = tf.get_variable(name="w_conv2", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv3 = tf.get_variable(name="w_conv3", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv4 = tf.get_variable(name="w_conv4", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv5 = tf.get_variable(name="w_conv5", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv6 = tf.get_variable(name="w_conv6", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv7 = tf.get_variable(name="w_conv7", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv8 = tf.get_variable(name="w_conv8", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv9 = tf.get_variable(name="w_conv9", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    w_conv10 = tf.get_variable(name="w_conv10", shape=[3, 3, 64, 1], initializer=tf.contrib.layers.xavier_initializer())

	# Biases
    b_conv1 = tf.Variable(tf.zeros([64]), name='b_conv1')
    b_conv2 = tf.Variable(tf.zeros([64]), name='b_conv2')
    b_conv3 = tf.Variable(tf.zeros([64]), name='b_conv3')
    b_conv4 = tf.Variable(tf.zeros([64]), name='b_conv4')
    b_conv5 = tf.Variable(tf.zeros([64]), name='b_conv5')
    b_conv6 = tf.Variable(tf.zeros([64]), name='b_conv6')
    b_conv7 = tf.Variable(tf.zeros([64]), name='b_conv7')
    b_conv8 = tf.Variable(tf.zeros([64]), name='b_conv8')
    b_conv9 = tf.Variable(tf.zeros([64]), name='b_conv9')
    b_conv10 = tf.Variable(tf.zeros([1]), name='b_conv10')

    shape_train = tf.Variable(SHAPE_TRAIN, name="SHAPE_TRAIN", dtype=tf.float32)

    # declaring inputs
    input_cnn = tf.placeholder(tf.float32)

    # implementing the network
    h_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(input_cnn, w_conv1, strides=[1,1,1,1], padding='SAME'),b_conv1))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4)
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
    h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, w_conv6, strides=[1,1,1,1], padding='SAME') + b_conv6)
    h_conv7 = tf.nn.relu(tf.nn.conv2d(h_conv6, w_conv7, strides=[1,1,1,1], padding='SAME') + b_conv7)
    h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, w_conv8, strides=[1,1,1,1], padding='SAME') + b_conv8)
    h_conv9 = tf.nn.relu(tf.nn.conv2d(h_conv8, w_conv9, strides=[1,1,1,1], padding='SAME') + b_conv9)
    h_conv10 = tf.nn.conv2d(h_conv9, w_conv10, strides=[1, 1, 1, 1], padding='SAME') + b_conv10

    # Loading the test input and the model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    for Ep in range(0, EPOCH_SIZE-1):
        MODEL_PATH = ALL_MODEL_PATH + 'tf' + str(Ep) + '.ckpt'
        TEST_SAVE_PATH = ALL_TEST_SAVE_PATH + 'Ep_' + str(Ep) + '/'
        saver.restore(sess, MODEL_PATH)
        print(glob.glob(TEST_PATH + '*.bmp'))
        for testImgName in glob.glob(TEST_PATH + '*.bmp'):
            print('Test Image %s'% path_leaf(testImgName))
            testImg = cv2.imread(testImgName, 0).astype(np.float32)
            testImg_normalized = testImg / 255
            test_input = np.array([testImg_normalized])
            test_elem = np.rollaxis(test_input, 0,3)
            test_data = test_elem[np.newaxis, ...]
            output_data = sess.run([h_conv10], feed_dict={input_cnn:test_data})
            output_image = output_data[0][0,:,:,0]
            output_image = output_image*255
            tst_name = path_leaf(testImgName)
            testedImgName = tst_name[0:-4] + '_Ep' + str(Ep) + '.bmp'
            cv2.imwrite(TEST_SAVE_PATH + testedImgName, output_image)

    print('Testing finished!')