## Training

This code is based on [LaMa](https://github.com/advimman/lama)

###  1. Prepare training and testing data

**Places dataset** 

```
# Download data from http://places2.csail.mit.edu/download.html
# Places365-Standard: Train(105GB)/Test(19GB)/Val(2.1GB) from High-resolution images section
wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar
wget http://data.csail.mit.edu/places/places365/val_large.tar
wget http://data.csail.mit.edu/places/places365/test_large.tar

# Unpack train/test/val data and create .yaml config for it
bash fetch_data/places_standard_train_prepare.sh
bash fetch_data/places_standard_test_val_prepare.sh

# Sample images for test and viz at the end of epoch
bash fetch_data/places_standard_test_val_sample.sh
bash fetch_data/places_standard_test_val_gen_masks.sh

# Run training
python3 bin/train.py -cn lama-fourier location=places_standard

# To evaluate trained model and report metrics as in our paper
# we need to sample previously unseen 30k images and generate masks for them
bash fetch_data/places_standard_evaluation_prepare_data.sh

# Infer model on thick/thin/medium masks in 256 and 512 and run evaluation 
# like this:
python3 bin/predict.py \
model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
indir=$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
outdir=$(pwd)/inference/random_thick_512 model.checkpoint=last.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
$(pwd)/inference/random_thick_512 \
$(pwd)/inference/random_thick_512_metrics.csv
```

**CelebA dataset** 

```
# Make shure you are in lama folder
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

# Download CelebA-HQ dataset
# Download data256x256.zip from https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P

# unzip & split into train/test/visualization & create config for it
bash fetch_data/celebahq_dataset_prepare.sh

# generate masks for test and visual_test at the end of epoch
bash fetch_data/celebahq_gen_masks.sh



# Infer model on thick/thin/medium masks in 256 and run evaluation 
# like this:
python3 bin/predict.py \
model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier-celeba_/ \
indir=$(pwd)/celeba-hq-dataset/visual_test_256/random_thick_256/ \
outdir=$(pwd)/inference/celeba_random_thick_256 model.checkpoint=last.ckpt
```

###  2. training

**2.1 Train on CelebA dataset** 

train DiffIR_s1

```
sh train_celebahqS1.sh
```

train DiffIR_s2

```
# convert pretrained model of DiffIR_s1
# modify the "path" item in S1forS2.py to the path of the checkpoint of DiffIR_S1 and obtain celeba-S1.pth
python3 S1forS2.py 
```
```
#set the "generatorS2_path" and "generatorS1_path" items of configs/training/DiffIRS2-celeba.yaml to the path of celeba-S1.pth
sh train_celebahqS2.sh
```

**2.2 Train on Place2-standard dataset** 

train DiffIR_s1

```
sh train_place256S1.sh
```

train DiffIR_s2

```
# convert pretrained model of DiffIR_s1
# modify the "path" item in S1forS2.py to the path of the checkpoint of DiffIR_S1 and obtain place-S1.pth
python3 S1forS2.py 
```
```
#set the "generatorS2_path" and "generatorS1_path" items of configs/training/DiffIRS2-place2.yaml to the path of place-S1.pth
sh train_place256S2.sh
```

**2.3 Train on Place2-Challenge dataset** 

train DiffIR_s1

```
sh train_place256_bigLdataS1.sh
```

train DiffIR_s2

```
# convert pretrained model of DiffIR_s1
# modify the "path" item in S1forS2.py to the path of the checkpoint of DiffIR_S1 and obtain placebigdata-S1.pth
python3 S1forS2.py 
```
```
#set the "generatorS2_path" and "generatorS1_path" items of configs/training/DiffIRbigdataS2-place2.yaml to the path of placebigdata-S1.pth
sh train_place256_bigLdataS2.sh
```


**Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Motion_Deblurring/Options/Deblurring_Restormer.yml](Options/Deblurring_Restormer.yml)

## Evaluation

Download the pre-trained [model](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK?usp=sharing) and place it in `./pretrained_models/`

#### Testing on GoPro dataset

- Download GoPro testset, run
```
python download_data.py --data test --dataset GoPro
```

- Testing
```
python test.py --dataset GoPro
```

#### Testing on HIDE dataset

- Download HIDE testset, run
```
python download_data.py --data test --dataset HIDE
```

- Testing
```
python test.py --dataset HIDE
```

#### Testing on RealBlur-J dataset

- Download RealBlur-J testset, run
```
python download_data.py --data test --dataset RealBlur_J
```

- Testing
```
python test.py --dataset RealBlur_J
```

#### Testing on RealBlur-R dataset

- Download RealBlur-R testset, run
```
python download_data.py --data test --dataset RealBlur_R
```

- Testing
```
python test.py --dataset RealBlur_R
```

#### To reproduce PSNR/SSIM scores of the paper (Table 2) on GoPro and HIDE datasets, run this MATLAB script

```
evaluate_gopro_hide.m 
```

#### To reproduce PSNR/SSIM scores of the paper (Table 2) on RealBlur dataset, run

```
evaluate_realblur.py 
```

This code is based on LaMa

Training, we take celeba-HQ as an example:

Pretrain DiffIR$_{S1}$:

sh train_celebahqS1.sh

在训练DiffIR$_{S2}$前，我们要将 S1forS2.py文件中的path设置为S1阶段预训练好模型的地址，然后设定save_path,再运行

python3 S1forS2.py

得到celeba-ta.pth

然后将configs/training/DiffIRS2-celeba.yaml中的generatorS2_path和generatorS1_path都设置为celeba-ta.pth的地址

训练DiffIR$_{S2}$:

sh train_celebahqS2.sh

*******测试******

设置test_celeba_256_thick.sh中的测设数据地址，以及模型地址，然后直接运行

sh test_celeba_256_thick.sh

可以得到运行的图像结果

然后运行,得到定量结果：

sh eval_celeba_256_thick.sh


