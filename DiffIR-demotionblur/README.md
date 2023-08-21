## Training

1. To download GoPro training and testing data, run
```
python download_data.py --data train-test
```

2. Generate image patches from full-resolution training images of GoPro dataset
```
python generate_patches_gopro.py 
```

3. To pretrain DiffIR_S1, run
```
sh trainS1.sh
```

4. To train DiffIR_S2, run
```
#set the 'pretrain_network_g' and 'pretrain_network_S1' in ./options/train_DiffIRS2.yml to be the path of DiffIR_S1's pre-trained model

sh trainS2.sh
```

**Note:** The above training script uses 8 GPUs by default. 

## Evaluation

Download the pre-trained [model](https://drive.google.com/drive/folders/1JWYaP9VVPX_Mh2w1Vezn74hck-oWSyMh?usp=drive_link) and place it in `./experiments/`

#### Testing on GoPro dataset

- Download GoPro testset, run
```
python download_data.py --data test --dataset GoPro
```

- Testing
```
# modify the dataset path in ./options/test_DiffIRS2.yml

sh test.sh 
```

#### Testing on HIDE dataset

- Download HIDE testset, run
```
python download_data.py --data test --dataset HIDE
```

- Testing
```
# modify the dataset path in ./options/test_DiffIRS2.yml

sh test.sh
```

#### To reproduce PSNR/SSIM scores of the paper on GoPro and HIDE datasets, run this MATLAB script

```
evaluate_gopro_hide.m 
```


