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
parser.add_argument("--sampling_mode", dest='sampling_mode', type=str, default='dense', help="specify sparse or dense sampling of patches to compte PieAPP")
parser.add_argument("--gpu_id", dest='gpu_id', type=str, default='7', help="specify which GPU to use (don't specify this argument if using CPU only)")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if args.sampling_mode == 'sparse':
	stride_val = 27
if args.sampling_mode == 'dense':
	stride_val = 6

def cal_pieapp(ref_path, A_path):

	imagesRef = np.expand_dims(cv2.imread(ref_path).astype('float32'),axis=0)
	imagesA = np.expand_dims(cv2.imread(A_path).astype('float32'),axis=0)
	_,rows,cols,ch = imagesRef.shape

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

	return score_accum/weight_accum

if __name__ == '__main__':

	# others
	# data_root = '/data1/liangjie/BasicSR_ALL/results/'
	# ref_root = '/data1/liangjie/BasicSR_ALL/datasets/'
	# ref_dirs = ['SISR_Test_matlab/Set5mod12', 'SISR_Test_matlab/Set14mod12', 'SISR_Test_matlab/Manga109mod12',
	# 			'SISR_Test_matlab/BSDS100mod12', 'SISR_Test_matlab/General100mod12', 'SISR_Test/Urban100',
	# 			'DIV2K/DIV2K_valid_HR/']
	# datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
	# img_dirs = ['SRGAN_official', 'ESRGAN_official', 'NatSR_official', 'USRGAN_official', 'SPSR_official', 'SPSR_DF2K',
	# 			'ESRGAN_ours_DIV2K', 'ESRGAN_ours_DIV2K_ema', 'ESRGAN_ours_DF2K', 'ESRGAN_ours_DF2K_ema']

	# SFTGAN
	data_root = '/data1/liangjie/BasicSR_ALL/results'
	ref_root = '/data1/liangjie/BasicSR_ALL/results/SFTGAN_official'
	ref_dirs = ['GT'] * 7
	datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
	img_dirs = ['SFTGAN_official']

	logoverall_path = '/data1/liangjie/BasicSR_ALL/results/table_logs/' + 'pieapp_last.txt'

	for index in range(len(ref_dirs)):
		ref_dir = os.path.join(ref_root, ref_dirs[index])
		for method in img_dirs:
			img_dir = os.path.join(data_root, method, datasets[index])

			log_path = '/data1/liangjie/BasicSR_ALL/results/table_logs/' + img_dir.replace('/', '_') + '_pieapp.txt'

			if not os.path.exists(log_path):

				img_list = sorted(glob.glob(os.path.join(img_dir, '*')))

				pieapp_all = []

				for i, img_path in enumerate(img_list):
					file_name = img_path.split('/')[-1]
					if 'DIV2K100' in img_dir and 'SFTGAN' not in img_dir:
						gt_path = os.path.join(ref_dir, file_name[:4] + '.png')
					elif 'Urban100' in img_dir and 'SFTGAN' not in img_dir:
						gt_path = os.path.join(ref_dir, file_name[:7] + '.png')
					elif 'SFTGAN' in img_dir:
						gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '_gt.png')
						if 'Urban100' in img_dir:
							gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '_' + file_name.split('_')[1] + '_gt.png')
					else:
						if '_' in file_name:
							gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '.png')
						else:
							gt_path = os.path.join(ref_dir, file_name)

					print(gt_path, img_path)
					pieapp = cal_pieapp(gt_path, img_path)
					pieapp_all.append(pieapp.item())

					log = f'{i + 1:3d}: {file_name:25}. \tpieapp: {pieapp.item():.6f}.'
					with open(log_path, 'a') as f:
						f.write(log + '\n')

				log = f'Average: pieapp_all: {sum(pieapp_all) / len(pieapp_all):.6f}'
				with open(log_path, 'a') as f:
					f.write(log + '\n')
				log_overall = method + '__' + datasets[index] + '__' + log
				with open(logoverall_path, 'a') as f:
					f.write(log_overall + '\n')
				print(log_overall)

				assert 0 == 1