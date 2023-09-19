import sys
sys.path.append('../')
from saliency import get_blk_attn
from utils import *
from plots import *
from run import *
from torchattacks import PGD as t_PGD
from torchattacks import FGSM as t_FGSM
import foolbox as fb

device = "cuda"

# model = vit_base_patch16_224(pretrained=False)
model = torch.load('/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth')
model = model.to(device)

# """
image_folder = "/home/raza.imam/Documents/HC701B/Project/adv_data/TB_adversarial_data/testing/Tuberculosis"
image_files = [
        f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")
    ]

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation((90,90)),
        transforms.CenterCrop(400),
        transforms.Resize((224, 224)),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]
)

image_path = os.path.join(image_folder, image_files[19])
image = Image.open(image_path)
img = transform(image)
img = img.unsqueeze(0)

eps=0.1

# ---Captum---
# pgd = PGD(model, lower_bound=0, upper_bound=1)
# pgd_image = pgd.perturb(img.cuda(), radius=0.13, step_size=eps, step_num=7, target=0) 
# pgd_img = torch.tensor((pgd_image.cpu().data.numpy()))

# fgsm = FGSM(model, lower_bound=0, upper_bound=1)
# fgsm_image = fgsm.perturb(img.cuda(), epsilon=eps, target=0)
# fgsm_img = torch.tensor((fgsm_image.cpu().data.numpy()))
# ---Captum---

# # ---Torchattack---
# images = img
# labels = torch.tensor([0])

# t_pgd = t_PGD(model, eps=eps, alpha=2/225, steps=100, random_start=True)
# pgd_img = t_pgd(images, labels)
# t_fgsm = t_FGSM(model, eps=eps)
# fgsm_img = t_fgsm(images, labels)
# # ---Torchattack--

# ---Foolbox---
f_model = fb.PyTorchModel(model, bounds=(0,1), device='cuda') #Foolbox's PGD
labels = torch.tensor([1])
f_pgd = fb.attacks.PGD()
_, pgd_img, success = f_pgd(f_model, img.cuda(), labels.cuda(), epsilons=eps)
f_fgsm = fb.attacks.FGSM()
_, fgsm_img, success = f_fgsm(f_model, img.cuda(), labels.cuda(), epsilons=eps)
# ---Foolbox---

img_attn = get_blk_attn(input_img=img.cuda(), blk=11, model=model)
pgd_img_attn = get_blk_attn(input_img=pgd_img.cuda(), blk=11, model=model)
fgsm_img_attn = get_blk_attn(input_img=fgsm_img.cuda(), blk=11, model=model)

plt.figure(figsize=(10, 10))
text = ["Original Image", "PGD Image", "FGSM Image", "Attn Clean", "Attn PGD", "Attn FGSM"]
for i, fig in enumerate([img.squeeze(0).permute(2,1,0).cpu(), pgd_img.squeeze(0).permute(2,1,0).cpu(), fgsm_img.squeeze(0).permute(2,1,0).cpu(), img_attn, pgd_img_attn, fgsm_img_attn]):
    print(fig.shape)
    plt.subplot(1, 6, i+1)
    plt.imshow(fig, cmap='inferno')
    plt.title(text[i])
plt.show()
# """

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from captum.attr import LayerActivation, LayerGradientXActivation
from captum.attr import visualization as viz
from tqdm import tqdm
from torchattacks import PGD as t_PGD
from torchattacks import FGSM as t_FGSM

# Define the image folder and load the model
image_folder = "/home/raza.imam/Documents/HC701B/Project/data/TB_data/testing/Tuberculosis/"
image_files = [
    f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")
]

# Define the transformation for image preprocessing
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation((90, 90)),
        transforms.CenterCrop(400),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Define the number of images to read (N)
N = 20  # Change this to the desired number of images

# Create an empty list to store the images
images_list = []

# Loop through the first N image files and apply the transformation
for i in range(N):
    image_path = os.path.join(image_folder, image_files[i])
    image = Image.open(image_path)
    img = transform(image)
    img = img.unsqueeze(0)
    images_list.append(img)

# Convert the list of images into a tensor
images_tensor = torch.cat(images_list, dim=0)

# Define epsilon values to iterate over
eps_list = [0, 0.003, 0.01, 0.02, 0.03, 0.1]

# Initialize lists to store average attention maps for each block and epsilon
avg_attentions = [[] for _ in range(12)]

# Loop through the images and epsilon values
for img in images_tensor:
    img = img.unsqueeze(0)

    # Initialize lists to store attention maps for each block and epsilon
    attentions = [[] for _ in range(12)]

    for eps in eps_list:
        images = img
        labels = torch.tensor([1])
        
        # pgd = PGD(model, lower_bound=0, upper_bound=1) #Captum's PGD
        # pgd_image = pgd.perturb(img.cuda(), radius=0.13, step_size=eps, step_num=7, target=0) 
        # pgd_img = torch.tensor((pgd_image.cpu().data.numpy()))
        
        # t_pgd = t_PGD(model, eps=eps, alpha=2/225, steps=100, random_start=True) #TorchAttack's PGD
        # pgd_img = t_pgd(images, labels)

        f_model = fb.PyTorchModel(model, bounds=(0,1), device='cuda') #Foolbox's PGD
        f_pgd = fb.attacks.PGD()
        _, pgd_img, success = f_pgd(f_model, images.cuda(), labels.cuda(), epsilons=eps)

        for block in range(12):
            img_attn = get_blk_attn(input_img=pgd_img.cuda(), blk=block, model=model)
            attentions[block].append(img_attn)

    # Calculate the average attention maps for each block and epsilon
    for block in range(12):
        avg_attention = np.mean(attentions[block], axis=0)
        avg_attentions[block].append(avg_attention)

# Plot the average attention maps
num_blocks = len(avg_attentions)
print(num_blocks)
num_epsilons = len(eps_list)
print(num_epsilons)

plt.figure(figsize=(20, 25))

for block in range(num_blocks):
    for eps_idx, epsilon in enumerate(eps_list):
        plt.subplot(num_blocks, num_epsilons, block * num_epsilons + eps_idx + 1)
        plt.imshow(avg_attentions[block][eps_idx], cmap='inferno')
        plt.title(f'Block {block}, Epsilon={epsilon}')

plt.tight_layout()
plt.show()
"""