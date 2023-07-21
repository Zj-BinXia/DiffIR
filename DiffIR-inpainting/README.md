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


