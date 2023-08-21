import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
import lpips
import argparse


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', type=str, default='/root/results/NTIRE2020-Track1')
    parser.add_argument('--folder_restored', type=str, default='/root/datasets/NTIRE2020-Track1/track1-valid-gt')
    args = parser.parse_args()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda(0)
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(args.folder_gt, '*.png')))
    lr_list = sorted(glob.glob(osp.join(args.folder_restored, '*.png')))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, (img_path, lr_path) in enumerate(zip(img_list,lr_list)):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(lr_path), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(0), img_gt.unsqueeze(0).cuda(0)).cpu().data.numpy()[0,0,0,0]
        # print(lpips_val)
        lpips_all.append(lpips_val)

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
