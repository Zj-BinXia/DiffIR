import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.utils import  img2tensor
from torch.utils import data as data
import cv2
from basicsr.utils.img_util import tensor2img
from DiffIR.archs.S2_arch import DiffIRS2
from basicsr.utils import  img2tensor
import argparse
from torch.nn import functional as F


def pad_test(lq,scale):
    if scale==1:
        window_size = 32
    elif scale==2:
        window_size = 16
    else:
        window_size = 8      
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = lq.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    lq = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return lq,mod_pad_h,mod_pad_w

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./experiments/DiffIRS2-GAN.pth')
    parser.add_argument('--im_path', type=str, default='/mnt/bn/xiabinpaint/dataset/NTIRE2020-Track1/track1-valid-input')
    parser.add_argument('--res_path', type=str, default='./outputs/')
    args = parser.parse_args()

    os.makedirs(args.res_path, exist_ok=True)
    model = DiffIRS2( n_encoder_res= 9, dim= 64, scale=args.scale,num_blocks= [13,1,1,1],num_refinement_blocks= 13,heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree")
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=True)
    model.to('cuda:0')
    model.eval()

    im_list = os.listdir(args.im_path)
    im_list.sort()
    im_list = [name for name in im_list if name.endswith('.png')]

    with torch.no_grad():
        for name in im_list:
            path = os.path.join(args.im_path, name)
            im = cv2.imread(path)
            im = img2tensor(im)
            im = im.unsqueeze(0).cuda(0)/255.
            lq,mod_pad_h,mod_pad_w= pad_test(im,args.scale)
            with torch.no_grad():
                sr = model(lq)
            _, _, h, w = sr.size()
            sr = sr[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]
            im_sr = tensor2img(sr, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
            save_path = os.path.join(args.res_path, name.split('.')[0]+'_out.png')
            cv2.imwrite(save_path, im_sr)
            print(save_path)
