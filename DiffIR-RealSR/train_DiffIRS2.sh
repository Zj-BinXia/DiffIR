CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3  -m torch.distributed.launch --nproc_per_node=8 --master_port=5685 DiffIR/train.py -opt options/train_DiffIRS2_x4.yml --launcher pytorch 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3  -m torch.distributed.launch --nproc_per_node=8 --master_port=5685 DiffIR/train.py -opt options/train_DiffIRS2_x2.yml --launcher pytorch 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3  -m torch.distributed.launch --nproc_per_node=8 --master_port=5685 DiffIR/train.py -opt options/train_DiffIRS2_x1.yml --launcher pytorch 
