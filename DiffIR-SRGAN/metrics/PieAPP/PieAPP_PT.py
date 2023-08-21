import numpy as np
import cv2
import sys
import torch
from torch.autograd import Variable
sys.path.append('model/')
from model.PieAPPv0pt1_PT import PieAPP
sys.path.append('utils/')
from utils.image_utils import *
import argparse
import os

######## check for model and download if not present
if not os.path.isfile('weights/PieAPPv0.1.pth'):
	print("downloading dataset")
	os.system("bash scripts/download_PieAPPv0.1_PT_weights.sh")
	if not os.path.isfile('weights/PieAPPv0.1.pth'):
		print("PieAPPv0.1.pth not downloaded")
		sys.exit()

######## variables
patch_size = 64
batch_size = 1

######## input args
parser = argparse.ArgumentParser()
parser.add_argument("--ref_path", dest='ref_path', type=str, default='/data1/liangjie/BasicSR_ALL/scripts/metrics/PieAPP/imgs/Ref.png', help="specify input reference")
parser.add_argument("--A_path", dest='A_path', type=str, default='/data1/liangjie/BasicSR_ALL/scripts/metrics/PieAPP/imgs/A.png', help="specify input image")
parser.add_argument("--sampling_mode", dest='sampling_mode', type=str, default='dense', help="specify sparse or dense sampling of patches to compte PieAPP")
parser.add_argument("--gpu_id", dest='gpu_id', type=str, default='3', help="specify which GPU to use")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

imagesA = np.expand_dims(cv2.imread(args.A_path),axis =0).astype('float32')
imagesRef = np.expand_dims(cv2.imread(args.ref_path),axis =0).astype('float32')
_,rows,cols,ch = imagesRef.shape

if args.sampling_mode == 'sparse':
	stride_val = 27
else:
	stride_val = 6

try:
    gpu_num = float(args.gpu_id)
    use_gpu = 1
except ValueError:
    use_gpu = 0
except TypeError:
    use_gpu = 0

y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val),np.array([rows - patch_size])), axis=0)
num_y = len(y_loc)
x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val),np.array([cols - patch_size])), axis=0)
num_x = len(x_loc)
num_patches_per_dim = 10
num_patches = 10


# state_dict = torch.load('weights/PieAPPv0.1.pth')
# for name, weights in state_dict.items():
#     print(name, weights.size())  # 可以查看模型中的模型名字和权重维度
#     if name == 'ref_score_subtract.weight': #判断需要修改维度的条件
#         state_dict[name] = weights.unsqueeze(0)  #去掉维度0，把(1,128)转为(128)
#        # print(name,weights.squeeze(0).size()) 查看转化后的模型名字和权重维度
# torch.save(state_dict, 'weights/PieAPPv0.1_.pth')

######## initialize the model
PieAPP_net = PieAPP(batch_size, num_patches_per_dim)
PieAPP_net.load_state_dict(torch.load('weights/PieAPPv0.1_.pth'))

if use_gpu == 1:
	PieAPP_net.cuda()

score_accum = 0.0
weight_accum = 0.0

# iterate through smaller size sub-images (to prevent memory overload)
for x_iter in range(0, -(-num_x//num_patches)):	
	for y_iter in range(0, -(-num_y//num_patches)):
		# compute the size of the subimage
		if (num_patches_per_dim*(x_iter + 1) >= num_x):				
			size_slice_cols = cols - x_loc[num_patches_per_dim*x_iter]
		else:
			size_slice_cols = x_loc[num_patches_per_dim*(x_iter + 1)] - x_loc[num_patches_per_dim*x_iter] + patch_size - stride_val			
		if (num_patches_per_dim*(y_iter + 1) >= num_y):
			size_slice_rows = rows - y_loc[num_patches_per_dim*y_iter]
		else:
			size_slice_rows = y_loc[num_patches_per_dim*(y_iter + 1)] - y_loc[num_patches_per_dim*y_iter] + patch_size - stride_val
		# obtain the subimage and samples patches
		A_sub_im = imagesA[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
		ref_sub_im = imagesRef[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
		A_patches, ref_patches = sample_patches(A_sub_im, ref_sub_im, patch_size=64, strideval=stride_val, random_selection=False, uniform_grid_mode = 'strided')
		num_patches_curr = A_patches.shape[0]/batch_size
		
		PieAPP_net.num_patches = num_patches_curr
		
		# initialize variable to be  fed to PieAPP_net
		A_patches_var = Variable(torch.from_numpy(np.transpose(A_patches,(0,3,1,2))), requires_grad=False)
		ref_patches_var = Variable(torch.from_numpy(np.transpose(ref_patches,(0,3,1,2))), requires_grad=False)
		if use_gpu == 1:
			A_patches_var = A_patches_var.cuda()
			ref_patches_var = ref_patches_var.cuda()

		# forward pass 
		_, PieAPP_patchwise_errors, PieAPP_patchwise_weights = PieAPP_net.compute_score(A_patches_var.float(), ref_patches_var.float())
		curr_err = PieAPP_patchwise_errors.cpu().data.numpy()	
		curr_weights = 	PieAPP_patchwise_weights.cpu().data.numpy()		
		score_accum += np.sum(np.multiply(curr_err, curr_weights))
		weight_accum += np.sum(curr_weights)

print('PieAPP value of '+args.A_path+ ' with respect to: '+str(score_accum/weight_accum))
