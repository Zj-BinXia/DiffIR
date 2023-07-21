CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
model.path=/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-lightv2/experiments/inpainting_2023-02-12_23-14-17_train_DiffIRS-celeba_/ \
indir=/mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_256/random_thin_256/ \
outdir=$(pwd)/inference/celeba_random_thin_256 model.checkpoint=last.ckpt