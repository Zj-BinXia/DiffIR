
# PieAPP dataset 
The dataset associated with the paper PieAPP: Perceptual Image Error Assessment through Pairwise Preference [[arXiv link](https://arxiv.org/abs/1806.02067)] can be downloaded from:
- [server containing a zip file with all data](https://web.ece.ucsb.edu/~ekta/projects/PieAPPv0.1/all_data_PieAPP_dataset_CVPR_2018.zip) (2.2GB),
- [Google Drive](https://drive.google.com/drive/folders/10RmBhfZFHESCXhhWq0b3BkO5z8ryw85p?usp=sharing) (ideal for quick browsing). 

The dataset contains undistorted high-quality reference images and several distorted versions of these reference images. Pairs of distorted images corresponding to a reference image are labeled with **probability of preference** labels. These labels that indicate the fraction of human population that considers one image to be visually closer to the reference over another in the pair. To ensure reliable pairwise probability of preference labels, we query 40 human subjects via Amazon Mechanical Turk for each image pair. Furthermore, we find the strategy of pairwise-preference labeling to be more robust to errors compared to traditional image-quality-labeling scheme based on mean opinion scores (MOS) - details about this and additional statistical analysis around reliable data collection can be found in the main paper and supplementary material. 

We make this dataset available for non-commercial and educational purposes only. 
The dataset contains a total of 200 undistorted reference images, divided into train / validation / test split.
These reference images are derived from the [Waterloo Exploration Dataset](https://ece.uwaterloo.ca/~k29ma/exploration/). We release the subset of 200 reference images used in PieAPP from the Waterloo Exploration Dataset with permissions for non-commercial, educational, use from the authors.
The users of the PieAPP dataset are requested to cite the Waterloo Exploration Dataset for the reference images, along with PieAPP dataset, as mentioned below.

## Dataset statistics
The training + validation set contain a total of 160 reference images and test set contains 40 reference images.
A total of 19,680 distorted images are generated for the train/val set and pairwise probability of preference labels for 77,280 image pairs are made available (derived from querying 40 human subjects for every pairwise comparison + ML estimation).

For test set, 15 distorted images per reference (total 600 distorted images) are created and dense pariwise comparisons (total 4200) are performed to label each image pair with a probability of preference, again derived from 40 human subjects' votes.

Overall, the PieAPP dataset provides a total of 20,280 distorted images derived from 200 reference images, and 81,480 pairwise probability-of-preference labels.

More details of dataset collection can be found in Sec.4 of the paper and supplementary document.

## Folder organization

- **reference_images** contains the undistorted referene images derived from the Waterloo Exploration dataset. 

- **distorted_images** contains all the distorted versions for each reference image (organized such that one folder contains distorted versions for one reference image), within the train / val / test sub-folders.

- **labels** contains csv files containing pairwise preference labels for distorted images (see main paper and supplementary material for details on how data is captured). There is one (two in case of test set) csv file for each reference image, in train / val / test subfolders.

## Interpreting the csv files contained in the labels folder

- For train and validation (val) set: there is one csv file for to each reference image (`ref_<image number>_pairwise_labels.csv`) containing pairwise labels:
        
        column 1: reference image
        column 2: distorted image A
        column 3: distorted image B
        column 4: raw probability of preference for image A, as obtained by MTurk data - not all pairs for a given reference are labeled; the ones that are not labeled are left blank in this column
        column 5: processed probability of preference for image A - we use the MTurk-labeled pairs to do an ML estimation of probability of preference for all pairs in a given inter set of distorted images for a reference (see section 4.3 in supplementary document for details)

- For the test set, each reference image has two csv files: one containing preference labels obtained through exhaustive per-reference pairwise comparisons using Amazon Mechanical Turk (naming convention: `ref_<image number>_pairwise_labels.csv`) 
and other containing per-image MAP-estimated scores for all distorted images (`ref_<image number>_per_image_score.csv`). 

        for ref_<image number>_pairwise_labels.csv in labels/test/ folder:
        column 1: reference image
        column 2: distorted image A
        column 3: distorted image B
        column 4: probability of preference for image A, as obtained by MTurk data - all pairs are labeled using human data, therefore no additional processing done to estimate missing pairs is needed

        for ref_<image number>_per_image_score.csv in labels/test/ folder:
        column 1: reference image
        column 2: distorted image A
        column 3: score for image A

Computing per-image scores enables evaluating the performance of image error/quality metrics using Pearson's Linear Correlation Coefficient and Spearman rank correlation coefficient. 
The per-image score indicates the **level of dissimilarity** of a given distorted image as compared to the reference image. That is, an image considered very different from the reference by humans would get a higher score.

Note that for the pairwise comparisons on test images, the reference image is also considered a "distorted" image and human pairwise preference between a distorted image and its reference image is also collected (again with 40 subjects). Since the ML-estimated scores using the Bradley-Terry model are correct up to an additive constant, this strategy allows for computing an MAP-estimated score for the reference image as well and serves as the constant which is then subtracted from the scores estimated for all the distorted versions of that reference image. As a result, the final reference-image score gets set to 0.

## Naming convention for images

Reference image: `ref_<image number>.png`

Distorted image: `distort_<image number>_<distortion type>_<inter or intra>_<identifier>.png`

For each reference image, several distorted versions are generated.
The name of any distorted image contains the following parts:
1. an image number that indicates its corresponding reference
2. the name of the distortion type
3. whether this distorted image is used during inter or intra type comparison (paper section 4.1) 
4. a unique identifier for a given distortion type: several realizations of any given distortion type are generated to choose from (for both inter-type and intra-type comparisons), this identifier helps distinguish those realizations

## Terms of Usage and how to cite this dataset
This dataset is made available only for non-commercial, educational purposes. The TERMS_OF_USE.pdf in the dataset directory highlights the details of the terms of usage.

If you find this dataset useful, please cite the PieAPP dataset:
        
        @InProceedings{Prashnani_2018_CVPR,
        author    = {Prashnani, Ekta and Cai, Hong and Mostofi, Yasamin and Sen, Pradeep},
        title     = {PieAPP: Perceptual Image-Error Assessment Through Pairwise Preference},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2018}
        }


Also, for the undistorted reference images, please cite the Waterloo Exploration dataset:
        
        @article{ma2017waterloo,
        author    = {Ma, Kede and Duanmu, Zhengfang and Wu, Qingbo and Wang, Zhou and Yong, Hongwei and Li, Hongliang and Zhang, Lei}, 
        title     = {{Waterloo Exploration Database}: New Challenges for Image Quality Assessment Models}, 
        journal   = {IEEE Transactions on Image Processing},
        volume    = {26},
        number    = {2},
        pages     = {1004--1016},
        month     = {Feb.},
        year      = {2017}
        }


For comments on improving this dataset release or questions or for reporting errors, please contact Ekta Prashnani or raise an issue on GitHub.

## Acknowledgements
This project was supported in part by NSF grants IIS-1321168 and IIS-1619376, as well as a Fall 2017 AI Grant (awarded to Ekta Prashnani).

