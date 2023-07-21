CUDA_VISIBLE_DEVICES=2 python3 bin/predict.py \
model.path=$(pwd)/experiments/DiffIRS2-celeba/ \
indir=/mnt/bn/xiabinpaint/dataset/celeba-hq-dataset/val_256/random_thick_256/ \
outdir=$(pwd)/inference/celeba_random_thick_256 model.checkpoint=last.ckpt