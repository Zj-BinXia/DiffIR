## DiffIR: Efficient diffusion model for image restoration (ICCV2023)

[Paper](https://arxiv.org/pdf/2303.09472.pdf) | [Project Page](https://github.com/Zj-BinXia/DiffIR) | [pretrained models](https://drive.google.com/drive/folders/10miVILiopE414GyaSZM3EFAZITeY9q0p?usp=sharing)

---

> **Abstract:** *Diffusion model (DM) has achieved SOTA performance by modeling the image synthesis process into a sequential application of a denoising network. However, different from image synthesis, image restoration (IR) has a strong constraint to generate results in accordance with ground-truth. Thus, for IR, traditional DMs running massive iterations on a large model to estimate whole images or feature maps is inefficient. To address this issue, we propose an efficient DM for IR (DiffIR), which consists of a compact IR prior extraction network (CPEN), dynamic IR transformer (DIRformer), and denoising network. Specifically, DiffIR has two training stages: pretraining and training DM. In pretraining, we input ground-truth images into CPEN$_{S1}$ to capture a compact IR prior representation (IPR) to guide DIRformer. In the second stage, we train the DM to directly estimate the same IRP as pretrained CPEN$_{S1}$ only using LQ images. We observe that since the IPR is only a compact vector,  DiffIR can use fewer iterations than traditional DM to obtain accurate estimations and generate more stable and realistic results. Since the iterations are few, our DiffIR can adopt a joint optimization of CPEN$_{S2}$, DIRformer, and denoising network, which can further reduce the estimation error influence. We conduct extensive experiments on several IR tasks and achieve SOTA performance while consuming less computational costs.* 
>
> <p align="center">
> <img width="800" src="figs/method.jpg">
> </p>

---


