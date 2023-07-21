This code is based on real-ESRGAN

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