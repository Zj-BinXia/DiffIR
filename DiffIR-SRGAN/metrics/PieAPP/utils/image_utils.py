import numpy as np
import sys
import cv2

def generate_images_from_list(A_names, B_names, ref_names, gt_labels, num_imgs_to_get, cursor, max_images):

	image_A = cv2.imread(A_names[cursor]) # sloppy

	batch_A = np.zeros((num_imgs_to_get,image_A.shape[0],image_A.shape[1],image_A.shape[2]))
	batch_B = np.zeros((num_imgs_to_get,image_A.shape[0],image_A.shape[1],image_A.shape[2]))
	batch_ref = np.zeros((num_imgs_to_get,image_A.shape[0],image_A.shape[1],image_A.shape[2]))
	batch_label = np.zeros((num_imgs_to_get,1))

	for n in range(0,num_imgs_to_get):
		batch_A[n,:,:,:] = cv2.imread(A_names[cursor]).astype('float32')
		# print B_names[cursor]
		batch_B[n,:,:,:] = cv2.imread(B_names[cursor]).astype('float32')
		batch_ref[n,:,:,:] = cv2.imread(ref_names[cursor]).astype('float32')
		batch_label[n] = gt_labels[cursor]
		cursor += 1
		if cursor == max_images:
			cursor = 0

	return batch_A, batch_B, batch_ref, batch_label, cursor

def sample_patches(batch_A, batch_ref, patch_size=None, patches_per_image=None, seed='', random_selection=True, uniform_grid_mode = 'strided',strideval = None): 

	# sampling modes:
	# 1. random selection (optionally with a seed value): samples patches_per_image number of patches randomly; required vars: patches_per_image, random_selection=True; optional: seed
	# 2. fixed sampling with stride on a uniform grid: select all patches from image with a stride of strideval;  required vars: strideVal, random_selection=False
	# 3. fixed sampling of patches_per_image patches on a uniform grid: patches from image on a sqrt(patches_per_image)xsqrt(patches_per_image) grid;  required vars: patches_per_image, random_selection=False

	num_rows = batch_A.shape[1]
	num_cols = batch_A.shape[2]
	num_channels = batch_A.shape[3]
	batch_size = batch_A.shape[0]

	if not random_selection:
		# sequentially select patches_per_image number of patches from ref and image A	
		if uniform_grid_mode == 'strided':
			temp_r = np.int_(np.floor(np.arange(0,num_rows-patch_size+1,strideval))) # patches_per_image is square-rootable
			temp_c = np.int_(np.floor(np.arange(0,num_cols-patch_size+1,strideval)))			
		else:
			temp_r = np.int_(np.floor(np.linspace(0,num_rows-patch_size+1,np.sqrt(patches_per_image)))) # patches_per_image is square-rootable
			temp_c = np.int_(np.floor(np.linspace(0,num_cols-patch_size+1,np.sqrt(patches_per_image)))) 		
		select_cols,select_rows = np.meshgrid(temp_c,temp_r)
		select_cols = np.reshape(select_cols,(select_cols.shape[0]*select_cols.shape[1],))
		select_rows = np.reshape(select_rows,(select_rows.shape[0]*select_rows.shape[1],))
		patches_per_image = select_rows.shape[0]

	# patch output pre-allocation
	patch_batch_A = np.zeros((batch_size*patches_per_image,patch_size,patch_size,num_channels))
	patch_batch_ref = np.zeros((batch_size*patches_per_image,patch_size,patch_size,num_channels))

	location = 0

	for iter_batch in range(0,batch_size):		
		if random_selection:            
			# randomly select patches_per_image number of patches
			if len(seed) > 0:
				np.random.seed(seed)
			select_rows = np.random.choice(num_rows-patch_size+1,patches_per_image)
			select_cols = np.random.choice(num_cols-patch_size+1,patches_per_image)

		for iter_patch in range(0,patches_per_image):
			# load batch A
			patch_batch_A[location:location+1,:,:,:] = batch_A[iter_batch,select_rows[iter_patch]:select_rows[iter_patch]+patch_size,
			select_cols[iter_patch]:select_cols[iter_patch]+patch_size,:]
			# load batch ref
			patch_batch_ref[location:location+1,:,:,:] = batch_ref[iter_batch,select_rows[iter_patch]:select_rows[iter_patch]+patch_size,
			select_cols[iter_patch]:select_cols[iter_patch]+patch_size,:]		
			location += 1
			
	return patch_batch_A, patch_batch_ref


