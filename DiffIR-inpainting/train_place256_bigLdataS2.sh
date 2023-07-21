
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 bin/train.py -cn DiffIRbigdataS2-place2 location=places_standard data.batch_size=15