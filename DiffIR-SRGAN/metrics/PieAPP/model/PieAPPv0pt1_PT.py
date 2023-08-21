import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PieAPP(nn.Module): # How to ensure that everything goes on a GPU? do I need to fetch?
	def __init__(self,batch_size,num_patches):
		super(PieAPP, self).__init__()
		self.conv1 = nn.Conv2d(3,64,3,padding=1)
		self.conv2 = nn.Conv2d(64,64,3,padding=1)
		self.pool2 = nn.MaxPool2d(2,2)
		self.conv3 = nn.Conv2d(64,64,3,padding=1)
		self.conv4 = nn.Conv2d(64,128,3,padding=1)
		self.pool4 = nn.MaxPool2d(2,2)
		self.conv5 = nn.Conv2d(128,128,3,padding=1)
		self.conv6 = nn.Conv2d(128,128,3,padding=1)
		self.pool6 = nn.MaxPool2d(2,2)
		self.conv7 = nn.Conv2d(128,256,3,padding=1)
		self.conv8 = nn.Conv2d(256,256,3,padding=1)
		self.pool8 = nn.MaxPool2d(2,2)
		self.conv9 = nn.Conv2d(256,256,3,padding=1)
		self.conv10 = nn.Conv2d(256,512,3,padding=1)
		self.pool10 = nn.MaxPool2d(2,2)
		self.conv11 = nn.Conv2d(512,512,3,padding=1)
		self.fc1_score = nn.Linear(120832, 512)
		self.fc2_score = nn.Linear(512,1)
		self.fc1_weight = nn.Linear(2048,512)
		self.fc2_weight = nn.Linear(512,1)
		self.ref_score_subtract = nn.Linear(1, 1)
		self.batch_size = batch_size
		self.num_patches = num_patches	

	def flatten(self,matrix): # takes NxCxHxW input and outputs NxHWC
		return matrix.view((self.batch_size*self.num_patches,-1))
	
	def compute_features(self,input):
		#conv1 -> relu -> conv2 -> relu -> pool2 -> conv3 -> relu
		x3 = F.relu(self.conv3(self.pool2(F.relu(self.conv2(F.relu(self.conv1(input))))))) 
		# conv4 -> relu -> pool4 -> conv5 -> relu		
		x5 = F.relu(self.conv5(self.pool4(F.relu(self.conv4(x3))))) 
		# conv6 -> relu -> pool6 -> conv7 -> relu		
		x7 = F.relu(self.conv7(self.pool6(F.relu(self.conv6(x5))))) 
		# conv8 -> relu -> pool8 -> conv9 -> relu		
		x9 = F.relu(self.conv9(self.pool8(F.relu(self.conv8(x7))))) 
		# conv10 -> relu -> pool10 -> conv11 -> relU
		x11 = self.flatten(F.relu(self.conv11(self.pool10(F.relu(self.conv10(x9)))))) 
		# flatten and concatenate
		feature_ms = torch.cat((self.flatten(x3),self.flatten(x5),self.flatten(x7),self.flatten(x9),x11),1) 
		return feature_ms, x11
	
	def compute_score(self,image_A_patches, image_ref_patches):
		A_multi_scale, A_coarse = self.compute_features(image_A_patches)
		ref_multi_scale, ref_coarse = self.compute_features(image_ref_patches)
		diff_ms = ref_multi_scale - A_multi_scale
		diff_coarse = ref_coarse - A_coarse		
		# per patch score: fc1_score -> relu -> fc2_score
		per_patch_score = self.ref_score_subtract(0.01*self.fc2_score(F.relu(self.fc1_score(diff_ms))))
		per_patch_score.view((-1,self.num_patches))
		# per patch weight: fc1_weight -> relu -> fc2_weight
		const = Variable(torch.from_numpy(0.000001*np.ones((1,))).float(), requires_grad=False) 
		const_cuda = const.cuda()		
		per_patch_weight = self.fc2_weight(F.relu(self.fc1_weight(diff_coarse)))+const_cuda
		per_patch_weight.view((-1,self.num_patches))
		product_val = torch.mul(per_patch_weight,per_patch_score)
		dot_product_val = torch.sum(product_val)
		norm_factor = torch.sum(per_patch_weight)
		return torch.div(dot_product_val, norm_factor), per_patch_score, per_patch_weight
