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

sh train_DiffIRS2_GAN.sh

or

sh train_DiffIRS2_GANv2.sh
```

**Note:** The above training script uses 8 GPUs by default. 




## Evaluation

Download the pre-trained [model](https://drive.google.com/drive/folders/1G3Ep0xd-uBpIXGZFdWzH1uVCOpJaqkOF?usp=drive_link) and place it in `./experiments/`

**Note**

- **DiffIRS2-GANx4**, **DiffIRS2-GANx2**, **DiffIRS2-GANx1** would have better fidelity.

- **DiffIRS2-GANx4-V2**, **DiffIRS2-GANx2-V2**, **DiffIRS2-GANx1-V2** would have better perceptual quality (including better denoising ability).

- V1 and V2 models have the same structure, and they can use the same evaluation and inference code.

- You should choose the V1 or V2 models according to your requirements.

#### Testing on NTIRE2020-Track1 dataset

Download  NTIRE2020 datasets [Link](https://competitions.codalab.org/competitions/22220)

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

#### To reproduce PSNR/LPIPS/DISTS scores of the paper

```
python3  Metric/PSNR.py --folder_gt PathtoGT  --folder_restored PathtoSR

python3  Metric/LPIPS.py --folder_gt PathtoGT  --folder_restored PathtoSR

python3  Metric/dists.py --folder_gt PathtoGT  --folder_restored PathtoSR
```

## Inference

```
python3  inference_diffir.py --im_path PathtoLR --res_path ./outputs --model_path Pathto4xModel --scale 4

python3  inference_diffir.py --im_path PathtoLR --res_path ./outputs --model_path Pathto2xModel --scale 2

python3  inference_diffir.py --im_path PathtoLR --res_path ./outputs --model_path Pathto1xModel --scale 1
```

## Finetuning

#### Step 1 Download pre-trained Model

Download the pre-trained [model](https://drive.google.com/drive/folders/1G3Ep0xd-uBpIXGZFdWzH1uVCOpJaqkOF?usp=drive_link) and place it in `./experiments/`

#### Step 2 Specify model Path

Specify the items "pretrain_network_S1", "pretrain_network_g", and "pretrain_network_d" items of finetune_DiffIRS2_GAN_x4_V2.yml to the downloaded pretrained model path.


#### Step 3 Prepare fine-tuning dataset

Deal your fine-tuning datasets as Dataset Preparation.

#### Step 4 Specify fine-tuning dataset Path

Specify the items "dataroot_gt" and "meta_info" items of finetune_DiffIRS2_GAN_x4_V2.yml to your GT datasets and the generated meta_info files.

#### Step 5 fine-tuning models

```
sh finetune_DiffIRS2_GANv2.sh
```



