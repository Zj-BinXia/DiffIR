CUDA_VISIBLE_DEVICES=5 python3 bin/predict.py \
model.path=/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-lightv2/experiments/inpainting_2023-02-14_22-21-15_train_DiffIRS-place2_/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/evaluation/random_thick_512/ \
outdir=$(pwd)/inference/random_thick_512 model.checkpoint=last.ckpt