CUDA_VISIBLE_DEVICES=2 python3 bin/predict.py \
model.path=/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-light/experiments/inpainting_2023-02-11_23-37-48_train_DiffIRT-celeba_/ \
indir=/mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_256/random_thick_256/ \
outdir=$(pwd)/inference/celeba_random_thick_256_T model.checkpoint=last.ckpt