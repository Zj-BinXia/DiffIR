
import torch

# epoch
# global_step
# pytorch-lightning_version
# state_dict
# callbacks
# optimizer_states
# lr_schedulers
# hparams_name
# hyper_parameters
# hparams_type

# path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/code-final/DiffIR-inpainting-final/experiments/Big-DiffIRS1-place/models/ori.ckpt"
# save_path ="/mnt/bn/xiabinpaint/ICCV-Inpainting/code-final/DiffIR-inpainting-final/experiments/Big-DiffIRS1-place/models/last.ckpt"

# path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/code-final/DiffIR-inpainting-final/experiments/DiffIRS1-celeba/models/ori.ckpt"
# save_path ="/mnt/bn/xiabinpaint/ICCV-Inpainting/code-final/DiffIR-inpainting-final/experiments/DiffIRS1-celeba/models/last.ckpt"

path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/code-final/DiffIR-inpainting-final/experiments/DiffIRS1-place/models/ori.ckpt"
save_path ="/mnt/bn/xiabinpaint/ICCV-Inpainting/code-final/DiffIR-inpainting-final/experiments/DiffIRS1-place/models/last.ckpt"

s=torch.load(path)


s["hyper_parameters"]["generator"]["kind"]="DiffIRS1"



torch.save(s,save_path)
