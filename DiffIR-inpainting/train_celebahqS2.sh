
# export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 bin/train.py -cn DiffIRT-celeba data.batch_size=4

export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 bin/train.py -cn DiffIRS2-celeba data.batch_size=4