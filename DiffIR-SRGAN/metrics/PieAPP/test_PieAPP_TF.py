import numpy as np
import tensorflow as tf
import cv2
import sys
sys.path.append('model/')
from model.PieAPPv0pt1_TF import PieAPP
import argparse
import os
import glob

######## check for model and download if not present
if not len(glob.glob('weights/PieAPP_model_v0.1.ckpt.*')) == 3:
	# print "downloading dataset"
	os.system("bash scripts/download_PieAPPv0.1_TF_weights.sh")
	if not len(glob.glob('weights/PieAPP_model_v0.1.ckpt.*')) == 3:
		# print "PieAPP_model_v0.1.ckpt files not downloaded"
		sys.exit()		

######## variables
patch_size = 64
batch_size = 1

######## input args
parser = argparse.ArgumentParser()
parser.add_argument("--ref_path", dest='ref_path', type=str, default='/data1/liangjie/BasicSR_ALL/scripts/metrics/PieAPP/imgs/Ref.png', help="specify input reference")
parser.add_argument("--A_path", dest='A_path', type=str, default='/data1/liangjie/BasicSR_ALL/scripts/metrics/PieAPP/imgs/A.png', help="specify input image")
parser.add_argument("--sampling_mode", dest='sampling_mode', type=str, default='dense', help="specify sparse or dense sampling of patches to compte PieAPP")
parser.add_argument("--gpu_id", dest='gpu_id', type=str, default='7', help="specify which GPU to use (don't specify this argument if using CPU only)")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

imagesRef = np.expand_dims(cv2.imread(args.ref_path).astype('float32'),axis=0)
imagesA = np.expand_dims(cv2.imread(args.A_path).astype('float32'),axis=0)
_,rows,cols,ch = imagesRef.shape
if args.sampling_mode == 'sparse':	
	stride_val = 27
if args.sampling_mode == 'dense':
	stride_val = 6
y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val),np.array([rows - patch_size])), axis=0)
num_y = len(y_loc)
x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val),np.array([cols - patch_size])), axis=0)
num_x = len(x_loc)
num_patches = 10

######## TF placeholder for graph input
image_A_batch = tf.placeholder(tf.float32)
image_ref_batch = tf.placeholder(tf.float32) #, [None, rows, cols, ch]

######## initialize the model
PieAPP_net = PieAPP(batch_size, args.sampling_mode)
PieAPP_value, patchwise_errors, patchwise_weights = PieAPP_net.forward(image_A_batch, image_ref_batch)
saverPieAPP = tf.train.Saver()

######## compute PieAPP
with tf.Session() as sess: 
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())	
	saverPieAPP.restore(sess, 'weights/PieAPP_model_v0.1.ckpt') # restore weights
	# iterate through smaller size sub-images (to prevent memory overload)
	score_accum = 0.0
	weight_accum = 0.0
	for x_iter in range(0, -(-num_x//num_patches)):
		for y_iter in range(0, -(-num_y//num_patches)):
			# compute scores on subimage to avoid memory issues
			# NOTE if image is 512x512 or smaller, PieAPP_value_fetched below gives the overall PieAPP value
			if (num_patches*(x_iter + 1) >= num_x):				
				size_slice_cols = cols - x_loc[num_patches*x_iter]
			else:
				size_slice_cols = x_loc[num_patches*(x_iter + 1)] - x_loc[num_patches*x_iter] + patch_size - stride_val			
			if (num_patches*(y_iter + 1) >= num_y):
				size_slice_rows = rows - y_loc[num_patches*y_iter]
			else:
				size_slice_rows = y_loc[num_patches*(y_iter + 1)] - y_loc[num_patches*y_iter] + patch_size - stride_val						
			im_A = imagesA[:, y_loc[num_patches*y_iter]:y_loc[num_patches*y_iter]+size_slice_rows, x_loc[num_patches*x_iter]:x_loc[num_patches*x_iter]+size_slice_cols,:]
			im_Ref = imagesRef[:, y_loc[num_patches*y_iter]:y_loc[num_patches*y_iter]+size_slice_rows, x_loc[num_patches*x_iter]:x_loc[num_patches*x_iter]+size_slice_cols,:]
			# forward pass 
			PieAPP_value_fetched, PieAPP_patchwise_errors, PieAPP_patchwise_weights = sess.run([PieAPP_value, patchwise_errors, patchwise_weights], 
				feed_dict={
				image_A_batch: im_A,
				image_ref_batch: im_Ref
				})
			score_accum += np.sum(np.multiply(PieAPP_patchwise_errors,PieAPP_patchwise_weights),axis=1)
			weight_accum += np.sum(PieAPP_patchwise_weights, axis=1)

	print('PieAPP value of '+args.A_path+ ' with respect to: '+str(score_accum/weight_accum))
	
