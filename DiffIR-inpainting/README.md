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


# To evaluate trained model and report metrics as in our paper
# we need to sample previously unseen 30k images and generate masks for them
bash fetch_data/places_standard_evaluation_prepare_data.sh

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


**Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify datasets path in configs/training
/location

## Evaluation

Download the pre-trained [model](https://drive.google.com/drive/folders/1RQXRWMqVaAsyyQt8T-3KtpS68ef8dh90?usp=drive_link) and place it in `./experiments/`

#### Testing on CelebA dataset

- Testing
```
sh test_celeba_256_thick.sh
```

- Calculating metric
```
sh eval_celeba_256_thick.sh
```

#### Testing on Place2-standard dataset


- Testing
```
sh test_place2_512_thick.sh
```

- Calculating metric
```
sh eval_place2_512_thick.sh
```

#### Testing on Place2-Challenge dataset


- Testing
```
sh test_place2_512_thick_big.sh
```

- Calculating metric
```
sh eval_place2_512_thick_big.sh
```








