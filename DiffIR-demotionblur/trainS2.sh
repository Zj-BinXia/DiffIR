#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4392 DiffIR/train.py -opt options/train_DiffIRS2.yml --launcher pytorch