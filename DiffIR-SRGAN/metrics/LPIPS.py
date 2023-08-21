import cv2
import os
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import argparse
from basicsr.utils import img2tensor


import lpips

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', type=str, default='/root/datasets/DIV2K100/HR')
    parser.add_argument('--folder_restored', type=str, default='/root/results/DIV2K100')
    args = parser.parse_args()

    img_list = sorted(glob.glob(osp.join(args.folder_gt, '*.png')))
    lr_list = sorted(glob.glob(osp.join(args.folder_restored, '*.png')))


    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, (img_path, lr_path) in enumerate(zip(img_list,lr_list)):

        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        _,h,w=img_gt.shape
        img_gt=img_gt[:,:h//4*4,:w//4*4]
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        # print(lpips_val)
        lpips_all.append(lpips_val.item())

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
