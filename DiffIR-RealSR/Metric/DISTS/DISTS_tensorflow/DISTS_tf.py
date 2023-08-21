# This is a tensorflow implementation of DISTS metric.
# Requirements: python >= 3.6, tensorflow-gpu >= 1.15

import tensorflow.compat.v1 as tf
import numpy as np
import time
import scipy.io as scio
from PIL import Image
import argparse
# tf.enable_eager_execution()
tf.disable_eager_execution()

class DISTS():
    def __init__(self):
        self.parameters = scio.loadmat('../weights/net_param.mat')
        self.chns = [3,64,128,256,512,512]
        self.mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1,1,1,3),name="img_mean")
        self.std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1,1,1,3),name="img_std")
        # self.alpha = tf.Variable(tf.random_normal(shape=(1,1,1,sum(self.chns)), mean=0.1, stddev=0.01),name="alpha")
        # self.beta = tf.Variable(tf.random_normal(shape=(1,1,1,sum(self.chns)), mean=0.1, stddev=0.01),name="beta")
        self.weights = scio.loadmat('../weights/alpha_beta.mat')
        self.alpha = tf.constant(np.reshape(self.weights['alpha'],(1,1,1,sum(self.chns))),name="alpha")
        self.beta = tf.constant(np.reshape(self.weights['beta'],(1,1,1,sum(self.chns))),name="beta")

    def get_features(self, img):

        x = (img - self.mean)/self.std

        self.conv1_1 = self.conv_layer(x, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.pool_layer(self.conv1_2, name="pool_1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.pool_layer(self.conv2_2, name="pool_2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.pool_layer(self.conv3_3, name="pool_3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.pool_layer(self.conv4_3, name="pool_4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

        return [img, self.conv1_2,self.conv2_2,self.conv3_3,self.conv4_3,self.conv5_3]

    def conv_layer(self, input, name):
        with tf.variable_scope(name) as _:
            filter = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, filter, strides=1, padding="SAME")
            bias = self.get_bias(name)
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return conv

    def pool_layer(self, input, name):
        # return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        with tf.variable_scope(name) as _:
            filter = tf.squeeze(tf.constant(self.parameters['L2'+name], name = "filter"),3)
            conv = tf.nn.conv2d(input**2, filter, strides=2, padding=[[0, 0], [1, 0], [1, 0], [0, 0]])
            return tf.sqrt(tf.maximum(conv, 1e-12))     

    def get_conv_filter(self, name):
        return tf.constant(self.parameters[name+'_weight'], name = "filter")

    def get_bias(self, name):
        return tf.constant(np.squeeze(self.parameters[name+'_bias']), name = "bias")

    def get_score(self, img1, img2):
        feats0 = self.get_features(img1)
        feats1 = self.get_features(img2)
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = tf.reduce_sum(self.alpha) + tf.reduce_sum(self.beta)
        alpha = tf.split(self.alpha/w_sum, self.chns, axis=3)
        beta = tf.split(self.beta/w_sum, self.chns, axis=3)
        for k in range(len(self.chns)):
            x_mean = tf.reduce_mean(feats0[k],[1,2], keepdims=True)
            y_mean = tf.reduce_mean(feats1[k],[1,2], keepdims=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+tf.reduce_sum(alpha[k]*S1, 3, keepdims=True)
            x_var = tf.reduce_mean((feats0[k]-x_mean)**2,[1,2], keepdims=True)
            y_var = tf.reduce_mean((feats1[k]-y_mean)**2,[1,2], keepdims=True)
            xy_cov = tf.reduce_mean(feats0[k]*feats1[k],[1,2], keepdims=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+tf.reduce_sum(beta[k]*S2, 3, keepdims=True)

        dist = 1-tf.squeeze(dist1+dist2)
        return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='../images/r0.png')
    parser.add_argument('--dist', type=str, default='../images/r1.png')
    args = parser.parse_args()
    model = DISTS()

    ref = np.array(Image.open(args.ref).convert("RGB"))
    ref = np.expand_dims(ref,axis=0)/255.
    dist = np.array(Image.open(args.dist).convert("RGB"))
    dist = np.expand_dims(dist,axis=0)/255.

    x = tf.placeholder(dtype=tf.float32, shape=ref.shape, name= "ref")
    y = tf.placeholder(dtype=tf.float32, shape=dist.shape, name= "dist")
    score = model.get_score(x,y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        score = sess.run(score, feed_dict={x: ref, y: dist})
        print(score)


