import sys
sys.path.append('../')
from saliency import get_blk_attn
from utils import *
from plots import *
from run import *

# """
# image_folder = "/home/raza.imam/Documents/HC701B/Project/adv_data/TB_adversarial_data/testing/Tuberculosis"
def return_a_img(image_folder, img_num=0):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(f".png")]
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
    print((image_files[0]))

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation((90,90)),
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

def return_npy_as_img(attns_folder, img_num=0):
    attn_files = [f for f in os.listdir(attns_folder) if f.endswith(f".npy")]
    attn_files = sorted(attn_files, key=lambda x: int(x.split('.')[0]))
    attn_path = os.path.join(attns_folder, attn_files[img_num])
    attn = np.load(attn_path, allow_pickle=True)
    return attn
"""
device = "cuda"
class_name = 'Normal' #Normal or TB class
eps = 0.03 #0.01, 0.03, 0.06
img_num =0
attack = 'PDG' #PDG or FGSM
succ_unsucc = 'Succ' #Check Succ or Unsucc samples

print(f"For '{class_name}', 'eps={eps}':--> Successful PGD+FGSM V/S Unsuccessful PGD+FGSM")

# model = vit_base_patch16_224(pretrained=False)
model = torch.load('/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth')
model = model.to(device)

#Imgs
succ_adv_folder = f"./data3/{attack}_{eps}/{class_name}/{succ_unsucc}/{succ_unsucc}_x/"
clean_folder = f"./data3/{attack}_{eps}/{class_name}/{succ_unsucc}/Clean_x/"
#Attns
succ_adv_folder_attn = f"./data3/{attack}_{eps}/{class_name}/{succ_unsucc}/{succ_unsucc}_x_attn/"
clean_folder_attn = f"./data3/{attack}_{eps}/{class_name}/{succ_unsucc}/Clean_x_attn/"

adv_img = return_a_img(succ_adv_folder, img_num=img_num)
cln_img = return_a_img(clean_folder, img_num=img_num)
# adv_attn = return_npy_as_img(attns_folder=succ_adv_folder_attn, img_num=img_num)
# cln_attn = return_npy_as_img(attns_folder=clean_folder_attn, img_num=img_num)

adv_attn = get_blk_attn(input_img=adv_img.cuda(), blk=-1, model=model)
cln_attn = get_blk_attn(input_img=cln_img.cuda(), blk=-1, model=model)

# pgd_img.squeeze(0).permute(2,1,0).cpu()

plt.figure(figsize=(10, 10))
text = ["Clean Image", "Successful Adv", "Attn Clean", "Attn Suc Adv"]
for i, fig in enumerate([cln_img.squeeze(0).permute(2,1,0).cpu(), adv_img.squeeze(0).permute(2,1,0).cpu(), cln_attn, adv_attn]):
    print(fig.shape)
    plt.subplot(1, 4, i+1)
    plt.imshow(fig, cmap='inferno')
    plt.title(text[i])
# plt.title(f"For '{class_name}', 'eps={epsilon}'")
plt.show()
"""


import matplotlib.pyplot as plt
import numpy as np

device = "cuda"
class_name = 'Normal'  # Normal or TB class
epsilons = [0.01, 0.03, 0.06]
attack = 'PDG'  # PDG or FGSM
img_num = 0
succ_unsucc = 'Succ' #Check Succ or Unsucc samples

model = torch.load('/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth')
model = model.to(device)

def get_attention_maps(input_img, model):
    attention_maps = []
    for blk in range(12):
        attn = get_blk_attn(input_img=input_img, blk=blk, model=model)
        attention_maps.append(attn)
    return attention_maps

fig, axes = plt.subplots(4, 13, figsize=(48, 24))  # Use 4 rows
clean_folder = f"./data3/{attack}_{epsilons[0]}/{class_name}/{succ_unsucc}/Clean_x/"
cln_img = return_a_img(clean_folder, img_num=img_num)
axes[3, 0].imshow(np.transpose(cln_img.squeeze().cpu().numpy(), (2, 1, 0)))  # Use the 4th row

for blk in range(12):
    cln_attn_maps = get_attention_maps(input_img=cln_img.cuda(), model=model)
    axes[3, blk+1].imshow(cln_attn_maps[blk], cmap='inferno')  # Use the 4th row
    axes[3, blk+1].set_title(f'Cln B:{blk}')

for i, epsilon in enumerate(epsilons):
    succ_adv_folder = f"./data3/{attack}_{epsilon}/{class_name}/{succ_unsucc}/{succ_unsucc}_x/"
    adv_img = return_a_img(succ_adv_folder, img_num=img_num)
    adv_attn_maps = get_attention_maps(input_img=adv_img.cuda(), model=model)
    axes[i, 0].imshow(np.transpose(adv_img.squeeze().cpu().numpy(), (2, 1, 0)))  # Use the 4th row

    for blk in range(12):
        axes[i, blk+1].imshow(adv_attn_maps[blk], cmap='inferno')
        axes[i, blk+1].set_title(f'E:{epsilon} B:{blk}')

for ax in axes.flat:
    ax.axis('off')

plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.05, hspace=0.05) 
plt.show()
