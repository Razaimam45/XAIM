import sys
sys.path.append('../')
from saliency import get_blk_attn
from utils import *
from plots import *
from run import *
import foolbox as fb

"""
# image_folder = "/home/raza.imam/Documents/HC701B/Project/adv_data/TB_adversarial_data/testing/Tuberculosis"
def return_a_img(image_folder, img_num=0):
    
    image_files = [
            f for f in os.listdir(image_folder) if f.endswith(f".png")
        ]
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
    print((image_files[0]))

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation((90,90)),
            # transforms.CenterCrop(400),
            transforms.Resize((224, 224)),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    )

    image_path = os.path.join(image_folder, image_files[img_num])
    image = Image.open(image_path)
    img = transform(image)
    img = img.unsqueeze(0)

    return img

device = "cuda"
class_name = 'Normal'
epsilon = 0.03
img_num = 0

print(f"For '{class_name}', 'eps={epsilon}':--> Successful PGD+FGSM V/S Unsuccessful PGD+FGSM")

# model = vit_base_patch16_224(pretrained=False)
model = torch.load('/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth')
model = model.to(device)

cln_img_folder = f"/home/raza.imam/Documents/XAIM/XAIM/data/clean/Test/{class_name}"
suc_pgd_img_folder = f"/home/raza.imam/Documents/XAIM/XAIM/data/successful_{epsilon}/PDG/Test/{class_name}"
suc_fgsm_img_folder = f"/home/raza.imam/Documents/XAIM/XAIM/data/successful_{epsilon}/FGSM/Test/{class_name}"
uns_pgd_img_folder = f"/home/raza.imam/Documents/XAIM/XAIM/data/unsuccessful_{epsilon}/PDG/Test/{class_name}"
uns_fgsm_img_folder = f"/home/raza.imam/Documents/XAIM/XAIM/data/unsuccessful_{epsilon}/FGSM/Test/{class_name}"

cln_img = return_a_img(cln_img_folder, img_num=img_num)
suc_pgd_img = return_a_img(suc_pgd_img_folder, img_num=img_num)
suc_fgsm_img = return_a_img(suc_fgsm_img_folder, img_num=img_num)
uns_pgd_img = return_a_img(uns_pgd_img_folder, img_num=img_num)
uns_fgsm_img = return_a_img(uns_fgsm_img_folder, img_num=img_num)

block = -1
cln_img_attn = get_blk_attn(input_img=cln_img.cuda(), blk=block, model=model)
suc_pgd_img_attn = get_blk_attn(input_img=suc_pgd_img.cuda(), blk=block, model=model)
suc_fgsm_img_attn = get_blk_attn(input_img=suc_fgsm_img.cuda(), blk=block, model=model)
uns_pgd_img_attn = get_blk_attn(input_img=uns_pgd_img.cuda(), blk=block, model=model)
uns_fgsm_img_attn = get_blk_attn(input_img=uns_fgsm_img.cuda(), blk=block, model=model)

# pgd_img.squeeze(0).permute(2,1,0).cpu()

plt.figure(figsize=(10, 10))
text = ["Clean Image", "Successful PGD", "Unuccessful PGD", "Attn Clean", "Attn Suc PGD", "Attn Uns PGD"]
for i, fig in enumerate([cln_img.squeeze(0).permute(2,1,0).cpu(), suc_pgd_img.squeeze(0).permute(2,1,0).cpu(), uns_pgd_img.squeeze(0).permute(2,1,0).cpu(), cln_img_attn, suc_pgd_img_attn, uns_pgd_img_attn]):
    print(fig.shape)
    plt.subplot(1, 6, i+1)
    plt.imshow(fig, cmap='inferno')
    plt.title(text[i])
# plt.title(f"For '{class_name}', 'eps={epsilon}'")
plt.show()
"""

# """
# image_folder = "/home/raza.imam/Documents/HC701B/Project/adv_data/TB_adversarial_data/testing/Tuberculosis"
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from captum.attr import LayerActivation, LayerGradientXActivation

model_path = '/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth'

model = torch.load(model_path)
model = model.to('cuda')

def return_a_img(image_folder, img_num=0):
    image_files = [
        f for f in os.listdir(image_folder) if f.endswith(".png")
    ]
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomRotation((90,90)),
            transforms.CenterCrop(200),
            transforms.Resize((224, 224)),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    )

    image_path = os.path.join(image_folder, image_files[img_num])
    image = Image.open(image_path)
    img = transform(image)
    img = img.unsqueeze(0)

    return img

device = "cuda"
class_name = 'Normal'

eps_list = [0.01, 0.03, 0.06]
img_num = 6
blocks = 12

plt.figure(figsize=(24, 8))

for eps_idx, epsilon in enumerate(eps_list):
    img_folder = f"/home/raza.imam/Documents/XAIM/XAIM/data/unsuccessful_{epsilon}/PDG/Test/{class_name}"
    img = return_a_img(img_folder, img_num=img_num)
    
    attentions = []

    for block in range(0, blocks):
        img_attn = get_blk_attn(input_img=img.cuda(), blk=block, model=model)
        attentions.append(img_attn.transpose(1,0))

    plt.subplot(3, 12, eps_idx + 1)
    plt.imshow(attentions[0], cmap='inferno')
    # plt.title(f'Epsilon={epsilon}, Block 1')
    
    for block in range(0, blocks):
        plt.subplot(3, 12, eps_idx * blocks + block + 1)
        plt.imshow(attentions[block], cmap='inferno')
        plt.title(f'B{block + 1}, E:{epsilon}')

plt.tight_layout()
plt.show()
# """
