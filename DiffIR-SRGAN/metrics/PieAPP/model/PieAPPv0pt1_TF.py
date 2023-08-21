from __future__ import with_statement
from __future__ import absolute_import
import sys
sys.path.append('../utils/')
import tensorflow as tf
import numpy as np
from utils.model_utils import *

class PieAPP(object):

	def __init__(self, batch_size,mode='sparse',istrain=False,keep_prob=None): # mode can be train or eval
		self.patch_size	= 64
		self.batch_size = batch_size
		self.mode = mode
		if istrain:
			self.KEEP_PROB = keep_prob
		else:
			self.KEEP_PROB = 1.0
		if mode == 'sparse':
			self.patch_stride = 27
		else: 
			self.patch_stride = 6
		self.init_vars()

	def init_vars(self):
		min_features = 64
		conv_init('conv1', 3, 3, 3, min_features)
		conv_init('conv2', min_features, 3, 3, min_features)
		conv_init('conv3', min_features, 3, 3, min_features)
		conv_init('conv4', min_features, 3, 3, min_features*2)
		conv_init('conv5', min_features*2, 3, 3, min_features*2)
		conv_init('conv6', min_features*2, 3, 3, min_features*2)
		conv_init('conv7', min_features*2, 3, 3, min_features*4)
		conv_init('conv8', min_features*4, 3, 3, min_features*4)
		conv_init('conv9', min_features*4, 3, 3, min_features*4)
		conv_init('conv10', min_features*4, 3, 3, min_features*8)
		conv_init('conv11', min_features*8, 3, 3, min_features*8)
		fc_init('fc1', 120832, 512)
		fc_init('fc2', 512, 1)
		fc_init('fc1w', 2048, 512)
		fc_init('fc2w', 512, 1)
		# the PieAPP value of the reference image with respect to itself is always a constant value and 
		# is subtracted from the value of the distorted image to obtain an image error along the error scale 
		# with origin at the reference PieAPP value.
		refscore = get_scope_variable('scores', 'refscore', shape = [1]) 
 
	def extract_features(self,image_patches):
		min_features = 64		
		# conv1
		conv1_A = conv(image_patches, 3, 3, min_features, 1, 1, padding = 'SAME', name = 'conv1')  
		# conv2
		conv2_A = conv(conv1_A, 3, 3, min_features, 1, 1, padding = 'SAME', name = 'conv2')
		pool2_A = max_pool(conv2_A, 2, 2, 2, 2, padding = 'SAME', name ='pool2') 
		# conv3
		conv3_A = conv(pool2_A, 3, 3, min_features, 1, 1, padding = 'SAME', name = 'conv3')
		shp = conv3_A.get_shape().as_list()
		f3A = tf.reshape(conv3_A,(-1,shp[1]*shp[2]*shp[3]))
		# conv4
		conv4_A = conv(conv3_A, 3, 3, min_features*2, 1, 1, padding = 'SAME', name = 'conv4')
		pool4_A = max_pool(conv4_A, 2, 2, 2, 2, padding = 'SAME', name ='pool4')
		# conv5
		conv5_A = conv(pool4_A, 3, 3, min_features*2, 1, 1, padding = 'SAME', name = 'conv5')
		shp = conv5_A.get_shape().as_list()
		f5A = tf.reshape(conv5_A,(-1,shp[1]*shp[2]*shp[3]))
		# conv6
		conv6_A = conv(conv5_A, 3, 3, min_features*2, 1, 1, padding = 'SAME', name = 'conv6')
		pool6_A = max_pool(conv6_A, 2, 2, 2, 2, padding = 'SAME', name ='pool6')  									
		# conv7
		conv7_A = conv(pool6_A, 3, 3, min_features*4, 1, 1, padding = 'SAME', name = 'conv7')
		shp = conv7_A.get_shape().as_list()
		f7A = tf.reshape(conv7_A,(-1,shp[1]*shp[2]*shp[3]))
		# conv8
		conv8_A = conv(conv7_A, 3, 3, min_features*4, 1, 1, padding = 'SAME', name = 'conv8')
		pool8_A = max_pool(conv8_A, 2, 2, 2, 2, padding = 'SAME', name ='pool8')      
 		# conv9
		conv9_A = conv(pool8_A, 3, 3, min_features*4, 1, 1, padding = 'SAME', name = 'conv9')
		shp = conv9_A.get_shape().as_list()
		f9A = tf.reshape(conv9_A, (-1,shp[1]*shp[2]*shp[3]))
		# conv10
		conv10_A = conv(conv9_A, 3, 3, min_features*8, 1, 1, padding = 'SAME', name = 'conv10')
		pool10_A = max_pool(conv10_A, 2, 2, 2, 2, padding = 'SAME', name ='pool10')      
 		# conv11
		conv11_A = conv(pool10_A, 3, 3, min_features*8, 1, 1, padding = 'SAME', name = 'conv11')
		shp = conv11_A.get_shape().as_list()
		f11A = tf.reshape(conv11_A,(-1,shp[1]*shp[2]*shp[3]))
		### flattening and concatenation		
		feature_A_multiscale = tf.concat([f3A, f5A, f7A, f9A, f11A],1) 
		return feature_A_multiscale, f11A

	def compute_score(self,diff_A_ref_ms, diff_A_ref_last):
		### scoring subnetwork image A and ref
		# fc1
		fc1_A_ref = fc(diff_A_ref_ms, diff_A_ref_ms.get_shape().as_list()[1], 512, name='fc1')
		dropout1_A_ref = dropout(fc1_A_ref, self.KEEP_PROB)
		# fc2
		fc2_A_ref = fc(dropout1_A_ref, 512, 1, name = 'fc2', relu=False)
		ref_score = get_scope_variable('scores', 'refscore', shape = [1])
		score_A_ref = tf.subtract(tf.multiply(tf.reshape(fc2_A_ref,[self.batch_size,-1]),tf.constant(0.01)),ref_score) # 0.01 is the sigmoid coefficient
		### weighing subnetwork image A and ref
		# fc1w
		fc1_A_ref_w = fc(diff_A_ref_last, diff_A_ref_last.get_shape().as_list()[1], 512, name='fc1w')
		dropout1_A_ref_w = dropout(fc1_A_ref_w, self.KEEP_PROB)
		# fc2w
		fc2_A_ref_w = fc(dropout1_A_ref_w, 512, 1, name = 'fc2w')
		weight_A_ref = tf.reshape(tf.add(fc2_A_ref_w,tf.constant(0.000001)),[self.batch_size,-1])	
		# final score computation: weighted average of scores
		norm_factor_A = tf.reduce_sum(weight_A_ref,axis=1)		
		product_score_weights_A = tf.multiply(weight_A_ref,score_A_ref)
		final_score_A = tf.divide(tf.reduce_sum(product_score_weights_A,axis=1),norm_factor_A)
		return final_score_A, score_A_ref, weight_A_ref

	def forward(self,image_A_batch,image_ref_batch):
		# patch extraction
		image_A_patches = extract_image_patches(image_A_batch, self.patch_size, self.patch_stride)
		image_ref_patches = extract_image_patches(image_ref_batch, self.patch_size, self.patch_stride)
		# feature extraction
		A_multiscale_feature, A_last_layer_feature = self.extract_features(image_A_patches)
		ref_multiscale_feature, ref_last_layer_feature = self.extract_features(image_ref_patches)
		# feature difference
		diff_A_ref_ms = tf.subtract(ref_multiscale_feature, A_multiscale_feature)
		diff_A_ref_last = tf.subtract(ref_last_layer_feature, A_last_layer_feature)
		# score computation
		score_A, per_patch_scores, per_patch_weights = self.compute_score(diff_A_ref_ms, diff_A_ref_last)
		return score_A, per_patch_scores, per_patch_weights