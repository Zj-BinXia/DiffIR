CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
model.path=$(pwd)/experiments/DiffIRS2-place/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/visual_test256_thick/ \
outdir=$(pwd)/inference/place2_visual_256 model.checkpoint=last.ckpt
