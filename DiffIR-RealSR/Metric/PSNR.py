import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
import lpips
import argparse
from basicsr.metrics import calculate_psnr, calculate_ssim


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', type=str, default='/root/results/NTIRE2020-Track1')
    parser.add_argument('--folder_restored', type=str, default='/root/datasets/NTIRE2020-Track1/track1-valid-gt')
    args = parser.parse_args()
    psnr_all = []
    ssim_all = []
    img_list = sorted(glob.glob(osp.join(args.folder_gt, '*.png')))
    lr_list = sorted(glob.glob(osp.join(args.folder_restored, '*.png')))
    for i, (img_path, lr_path) in enumerate(zip(img_list,lr_list)):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_restored = cv2.imread(osp.join(lr_path), cv2.IMREAD_UNCHANGED)
        psnr=calculate_psnr(img_restored, img_gt, crop_border=4, test_y_channel=True)
        ssim=calculate_ssim(img_restored, img_gt, crop_border=4, test_y_channel=True)
        psnr_all.append(psnr)
        ssim_all.append(ssim)

    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f}')
    print(f'Average: SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    main()
