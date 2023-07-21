CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
model.path=/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-lightv2/experiments/inpainting_2023-02-14_22-21-15_train_DiffIRS-place2_/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/visual_test256_thick/ \
outdir=$(pwd)/inference/place2_visual_256 model.checkpoint=last.ckpt