CUDA_VISIBLE_DEVICES=7 python3 bin/predict.py \
model.path=$(pwd)/experiments/Big-DiffIRS2-place/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/evaluation/random_thin_512/ \
outdir=$(pwd)/inference/random_thin_512_big model.checkpoint=last.ckpt
