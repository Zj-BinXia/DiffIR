CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \
model.path=$(pwd)/experiments/DiffIRS2-celeba/ \
indir=/mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_256/random_thin_256/ \
outdir=$(pwd)/inference/celeba_random_thin_256 model.checkpoint=last.ckpt
