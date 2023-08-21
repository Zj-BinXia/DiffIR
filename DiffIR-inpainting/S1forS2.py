import torch
path = "./experiments/DiffIRS1-celeba/models/last.ckpt"
save_path = "./celeba-S1.pth"
#path = "./experiments/DiffIRS1-place/models/last.ckpt"
#save_path = "./place-S1.pth"
#path = "./experiments/Big-DiffIRS1-place/models/last.ckpt"
#save_path = "./placebigdata-S1.pth"
s=torch.load(path)
# for k,v in s.items():
#    print(k)

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
