## Training

### 1. Dataset Preparation

We use DF2K (DIV2K and Flickr2K) + OST datasets for our training. Only HR images are required. <br>
You can download from :

1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3. OST: https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip

Here are steps for data preparation.

#### Step 1: [Optional] Generate multi-scale images

For the DF2K dataset, we use a multi-scale strategy, *i.e.*, we downsample HR images to obtain several Ground-Truth images with different scales. <br>
You can use the [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py) script to generate multi-scale images. <br>
Note that this step can be omitted if you just want to have a fast try.

```bash
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/DF2K_HR --output datasets/DF2K/DF2K_multiscale
```

#### Step 2: [Optional] Crop to sub-images

We then crop DF2K images into sub-images for faster IO and processing.<br>
This step is optional if your IO is enough or your disk space is limited.

You can use the [scripts/extract_subimages.py](scripts/extract_subimages.py) script. Here is the example:

```bash
 python scripts/extract_subimages.py --input datasets/DF2K/DF2K_multiscale --output datasets/DF2K/DF2K_multiscale_sub --crop_size 400 --step 200
```

#### Step 3: Prepare a txt for meta information

You need to prepare a txt file containing the image paths. The following are some examples in `meta_info_DF2Kmultiscale+OST_sub.txt` (As different users may have different sub-images partitions, this file is not suitable for your purpose and you need to prepare your own txt file):

```txt
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

You can use the [scripts/generate_meta_info.py](scripts/generate_meta_info.py) script to generate the txt file. <br>
You can merge several folders into one meta_info txt. Here is the example:

```bash
 python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR datasets/DF2K/DF2K_multiscale --root datasets/DF2K datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```

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

Download the pre-trained [model](https://drive.google.com/drive/folders/1G3Ep0xd-uBpIXGZFdWzH1uVCOpJaqkOF?usp=drive_link) and place it in `./experiments/`

#### Testing on NTIRE2020-Track1 dataset

- Testing
```
# modify the dataset path in ./options/test_DiffIRS2_GAN_x4.yml
sh test.sh 
```

#### Testing on AIM2019-Track2 dataset

- Testing
```
# modify the dataset path in ./options/test_DiffIRS2_GAN_x4.yml
sh test.sh 
```

#### Testing on RealSRSet dataset

Download Canon datasets Link: [Google Drive](https://drive.google.com/open?id=17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM), [Baidu Drive](https://pan.baidu.com/s/1dn4q-7E2_iJkNXx4MPdVng)(code: 2n93)

- Testing
```
# modify the dataset path in ./options/test_DiffIRS2_GAN_x4.yml
sh test.sh 
```

#### To reproduce PSNR/SSIM scores of the paper on GoPro and HIDE datasets, run this MATLAB script

```
evaluate_gopro_hide.m 
```





**数据集

制作流程详见real-esrgan,把DF2K预先切块，可以加快读取速度

**Training:

先跑
sh train_DiffIRS1.sh

得到最后的模型命名为DiffIRS1.pth，然后将这个模型的地址贴到
options/train_DiffIRS2_x4.yml的pretrain_network_g与 pretrain_network_S1去

然后运行
sh train_DiffIRS2.sh

然后将模型命名为DiffIRS2.pth，然后将这个模型的地址贴到
/options/train_DiffIRS2_GAN_x4.yml的pretrain_network_g上去

然后运行
sh train_DiffIRS2_GAN.sh

**Testing:

sh test.sh

指标测试工具，PSNR,SSIM,LPIPS 我用的是MM-realSR里提供的工具
