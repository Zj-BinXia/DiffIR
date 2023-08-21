## Training

### 1. Dataset Preparation

Please prepare the training dataset following the instructions in [BasicSR](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md).

### 2.  Pretrain DiffIR_S1
```
sh trainS1.sh
```

### 3.  Train DiffIR_S2

```
#set the 'pretrain_network_g' and 'pretrain_network_S1' in ./options/train_DiffIRS2_x4.yml to be the path of DiffIR_S1's pre-trained model

sh trainS2.sh
```

### 4.  Train DiffIR_S2_GAN

```
#set the 'pretrain_network_g' and 'pretrain_network_S1' in ./options/train_DiffIRS2_GAN_x4.yml to be the path of DiffIR_S2 and DiffIR_S1's trained model, respectively.

sh trainS2.sh
```

**Note:** The above training script uses 8 GPUs by default. 

## Evaluation

Download the pre-trained [model](https://drive.google.com/drive/folders/1Mmhz6Sx9tz-n3QJAd6w-UlxdugTEH2fV?usp=drive_link) and place it in `./experiments/`

#### Testing on datasets

Please download the testing datasets from  [GoogleDrive](https://drive.google.com/drive/folders/1dcOxsgkJPfGrwzQ4DJhXOx3DRyANznZN?usp=sharing) (or [Baidu Disk](https://pan.baidu.com/s/1V6oqQaHpKm3LJkmWufV7nA?pwd=u34j) or [OneDrive](https://connectpolyu-my.sharepoint.com/:u:/g/personal/19109963r_connect_polyu_hk/EX7RLNhUYa5CjoyeuawL55MBWA_wQfy7d5e4bGaz3BcVuQ?e=odp7My)).

- Testing
```
# modify the dataset path in ./options/test_DiffIRS2_GAN_x4.yml 

sh test.sh 
```



#### To reproduce PSNR/LPIPS/DISTS scores of the paper

```
python3  Metric/PSNR.py --folder_gt PathtoGT  --folder_restored PathtoSR

python3  Metric/LPIPS.py --folder_gt PathtoGT  --folder_restored PathtoSR

python3  Metric/dists.py --folder_gt PathtoGT  --folder_restored PathtoSR
```
