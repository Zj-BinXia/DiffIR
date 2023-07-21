CUDA_VISIBLE_DEVICES=7 python3 bin/predict.py \
model.path=/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-lightv2/experiments/inpainting_2023-02-23_00-38-26_train_DiffIRbigS-place2_/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/evaluation/random_thin_512/ \
outdir=$(pwd)/inference/random_thin_512_big model.checkpoint=last.ckpt