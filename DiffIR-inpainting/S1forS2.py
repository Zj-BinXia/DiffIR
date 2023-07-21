import torch
# path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-light/experiments/inpainting_2023-02-11_23-37-48_train_DiffIRT-celeba_/models/last.ckpt"
# path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-light/experiments/inpainting_2023-02-10_11-28-11_train_DiffIRT-place2_/models/last.ckpt"
path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-lightv2/experiments/inpainting_2023-02-19_23-03-29_train_DiffIRbigT-place2_/models/last.ckpt"
save_path = "/mnt/bn/xiabinpaint/ICCV-Inpainting/KDSR-inpainting-lightv2/placebig-ta.pth"
s=torch.load(path)
for k,v in s.items():
    print(k)

print("***************")
# for k,v in s["state_dict"].items():
#     if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
#         print(k)
# print(s["state_dict"])
new={}
for k,v in s["state_dict"].items():
    if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
        k=k[10:]
        print(k)
        new[k]=v

for k,v in new.items():
    print(k)
torch.save(new,save_path)