CUDA_VISIBLE_DEVICES=3 python3 bin/predict.py \
model.path=$(pwd)/experiments/DiffIRS2-place/ \
indir=/mnt/bn/xiabinpaint/dataset/inpainting/place2/evaluation/random_thin_512/ \
outdir=$(pwd)/inference/random_thiin_512 model.checkpoint=last.ckpt
