CUDA_VISIBLE_DEVICES=6 python3 bin/predict.py \
model.path=$(pwd)/experiments/Big-DiffIRS2-place/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/evaluation/random_thick_512/ \
outdir=$(pwd)/inference/random_thick_512_big model.checkpoint=last.ckpt
