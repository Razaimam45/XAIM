# %% [markdown]
# ## Filter Out Successful Advarsarial Samples

# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import PIL
import argparse

# %%
model_path = "../vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth"

# %%
def get_model(model_path, device):
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    return model

# %%
transform = transforms.Compose(
[
    transforms.Grayscale(num_output_channels=3),
    # transforms.RandomRotation((90,90)),
    # transforms.CenterCrop(400),
    transforms.Resize((224, 224)),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

def get_blk_attn(input_img, blk, model, patch_size=16): #REVIEW:function to get mean attention of an image for a blk.
    # a dict to store the activations
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    h = model.blocks[blk].attn.attn_drop.register_forward_hook(getActivation("attn"))

    model.eval()
    out = model(input_img)
    
    attentions = activation['attn']
    nh = attentions.shape[1]
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    w_featmap = input_img.shape[-2] // patch_size
    h_featmap = input_img.shape[-1] // patch_size

    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    attentions = nn.functional.interpolate(attentions.unsqueeze(
            0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = attentions.transpose(0,2,1)
    # print(attentions.shape) #REVIEW:list of attentions from each head, after interpolation. Shape = torch.Size([1, 12, 224, 224])
    mean_attention = np.mean(attentions, 0)

    return mean_attention, torch.argmax(out).item() #REVIEW:Return mean of 12 head attentions


# %%
classes = ["Normal", "Tuberculosis"]
device = 'cuda:3'
data_path = '../data/TB_data'

# %%
model = get_model(model_path=model_path, device=device)


normal_testing = [f for f in os.listdir(os.path.join(data_path, 'testing', classes[0])) if f.endswith(".jpg") or f.endswith(".png")]
tb_testing = [f for f in os.listdir(os.path.join(data_path, 'testing', classes[1])) if f.endswith(".jpg") or f.endswith(".png")]
normal_traning = [f for f in os.listdir(os.path.join(data_path, 'training', classes[0])) if f.endswith(".jpg") or f.endswith(".png")]
tb_traning = [f for f in os.listdir(os.path.join(data_path, 'training', classes[1])) if f.endswith(".jpg") or f.endswith(".png")]


img = PIL.Image.open(os.path.join(data_path, 'training', classes[0], normal_traning[100]))
img = transform(img).unsqueeze(0).to(device)
pred = model(img)
print(pred, normal_traning[3])


img = PIL.Image.open(os.path.join(data_path, 'training', classes[1], tb_traning[100]))
img = transform(img).unsqueeze(0).to(device)
pred = model(img)
pred, tb_traning[0]

blk = -1

successfull_data_path = '../data/TB_data/successfull_samples'
os.makedirs(os.path.join(successfull_data_path, 'training', "Normal"), exist_ok=True)
os.makedirs(os.path.join(successfull_data_path, 'training', "Tuberculosis"), exist_ok=True)
os.makedirs(os.path.join(successfull_data_path, 'testing', "Normal"), exist_ok=True)
os.makedirs(os.path.join(successfull_data_path, 'testing', "Tuberculosis"), exist_ok=True)

# %%
from captum.robust import PGD, FGSM
import foolbox as fb


# %%
def apply_pdg(f_pgd, f_model, images, device="cuda:3", eps=0.03, radius = 0.13, step_num=40, labels=torch.tensor([1]), target=0, pgd = None, attack_lib='Foolbox'):
    # if pgd is None:
    #     pgd = PGD(model, lower_bound=0, upper_bound=1)

    # labels = torch.tensor([1])
    adv_imgs = []
    for input_img in tqdm(images):
        input_img = input_img.float()
        if attack_lib=='Foolbox':
            _, perturbed_image, success = f_pgd(f_model, input_img.to(device), labels.to(device), epsilons=eps)
        else:
            perturbed_image = pgd.perturb(input_img.to(device), radius=radius, step_size=eps, step_num=step_num, target=target) #step_size = epsilon in PGD case
        adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
        
        adv_imgs.append(adv_img.squeeze(0))
    adv_imgs = torch.stack(adv_imgs)
    return adv_imgs


def apply_fgsm(images, device="cuda:3", eps=0.03, attack_lib="Foolbox", labels = torch.tensor([1]), fgsm = None, f_fgsm = None, f_model = None):    
    adv_imgs = []
    for input_img in tqdm(images):
        input_img = input_img.float()
        if attack_lib=='Foolbox':
            _, perturbed_image, success = f_fgsm(f_model, input_img.to(device), labels.to(device), epsilons=eps)
        else:
            perturbed_image = fgsm.perturb(input_img.to(device), epsilon=eps, target=0) 
        adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
        adv_imgs.append(adv_img.squeeze(0))
    adv_imgs = torch.stack(adv_imgs)
    return adv_imgs


print('loading the images')

# %%
image_test_normal = []
for img_path in tqdm(normal_testing):
    img = PIL.Image.open(os.path.join(data_path, 'testing', classes[0], img_path))
    img = transform(img).unsqueeze(0).to(device)
    image_test_normal.append(img)


image_test_tb = []
for img_path in tqdm(tb_testing):
    img = PIL.Image.open(os.path.join(data_path, 'testing', classes[1], img_path))
    img = transform(img).unsqueeze(0).to(device)
    image_test_tb.append(img)


images_train_normal = []
for img_path in tqdm(normal_traning):
    img = PIL.Image.open(os.path.join(data_path, 'training', classes[0], img_path))
    img = transform(img).unsqueeze(0).to(device)
    images_train_normal.append(img)


images_train_tb = []
for img_path in tqdm(tb_traning):
    img = PIL.Image.open(os.path.join(data_path, 'training', classes[1], img_path))
    img = transform(img).unsqueeze(0).to(device)
    images_train_tb.append(img)

# %%
# adv_images_test_normal = apply_pdg(model, image_test_normal)

# %%
def get_successful_adv_images (images_list, cls_idx):
    selected_imgs = []
    # selected_imgs_name = []
    attns = []
    for img in tqdm(images_list):
        attn, pred = get_blk_attn(img.unsqueeze(0).to(device), blk, model)
        if pred != cls_idx: # check if pred is not correct
            selected_imgs.append(img.permute(1,2,0))
            # selected_imgs_name.append(img_path)
            attns.append(attn)
            # break
            # img.save(os.path.join(data_path, 'testing', classes[pred_i], img_path[:-4]+'.png'))
            # break
    # return selected_imgs_name, selected_imgs, attns
    return selected_imgs, attns

# -------------------------------------- # 
# -------------------------------------- #
# -------------------------------------- #


# # argparse
# parser = argparse.ArgumentParser(description='Get successful adversarial samples')
# parser.add_argument('--eps', type=float, default=0.03, help='epsilon for PGD')
# args = parser.parse_args()
# eps = args.eps

print("----------------------------------------------")
print("----------------------------------------------")
# print(f'Running for Epsilon = {eps}')
print("----------------------------------------------")
print("----------------------------------------------")

# %%
pgd = PGD(model, lower_bound=0, upper_bound=1)
f_model = fb.PyTorchModel(model, bounds=(0,1), device=device) #Foolbox's PGD
f_pgd = fb.attacks.PGD()

fgsm = FGSM(model, lower_bound=0, upper_bound=1)
f_model = fb.PyTorchModel(model, bounds=(0,1), device=device) #Foolbox's PGD
f_fgsm = fb.attacks.FGSM()

for eps in [0.03, 0.06, 0.01]:
    for attack in ['PDG', 'FGSM']:
        pred_i = 0 # 0 --> Normal, 1 --> TB
        print(classes[pred_i])
        print(f'Running {attack} for Epsilon = {eps} on Test normal')
        if attack == 'PDG':
            adv_images_test_normal = apply_pdg(images=  image_test_normal, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
        else:
            adv_images_test_normal = apply_fgsm(images=  image_test_normal, f_model=f_model, fgsm=fgsm, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=device, eps=eps)
        selected_imgs, attns = get_successful_adv_images(adv_images_test_normal, pred_i)

        save_folder = f"./successful_{eps}/{attack}/Test/Normal"
        save_folder_attn = f"./successfull_attn_{eps}/{attack}/Test/Normal"
        print(f'Saving {len(selected_imgs)} images to {save_folder}')
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_attn, exist_ok=True)
        for i, img in enumerate(selected_imgs):
            plt.imsave(os.path.join(save_folder, f"{i}.png"), np.array(img))
            np.save(os.path.join(save_folder_attn, f"{i}.npy"), attns[i])


        # Test TB
        pred_i = 1 # 0 --> Normal, 1 --> TB
        print(classes[pred_i])
        print(f'Running {attack} for Epsilon = {eps} on Test TB')
        if attack == 'PDG':
            adv_images_test_normal = apply_pdg(images=image_test_tb, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)

        else:
            adv_images_test_normal = apply_fgsm(images=image_test_tb, f_model=f_model, fgsm=fgsm, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=device, eps=eps)

        # for tuberculosis
        selected_imgs, attns = get_successful_adv_images(adv_images_test_normal, pred_i)


        save_folder = f"./successful_{eps}/{attack}/Test/TB"
        save_folder_attn = f"./successfull_attn_{eps}/{attack}/Test/TB"
        print(f'Saving {len(selected_imgs)} images to {save_folder}')

        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_attn, exist_ok=True)

        for i, img in enumerate(selected_imgs):
            plt.imsave(os.path.join(save_folder, f"{i}.png"), np.array(img))
            np.save(os.path.join(save_folder_attn, f"{i}.npy"), attns[i])


        # train normal

        pred_i = 0 # 0 --> Normal, 1 --> TB
        print(classes[pred_i])
        print(f'Running {attack} for Epsilon = {eps} on Train normal')
        if attack == 'PDG':
            adv_images_test_normal = apply_pdg(images=images_train_normal, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
        else:
            adv_images_test_normal = apply_fgsm(images=images_train_normal, f_model=f_model, fgsm=fgsm, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=device, eps=eps)

        selected_imgs, attns = get_successful_adv_images(adv_images_test_normal, pred_i)
        save_folder = f"./successful_{eps}/{attack}/Train/Normal"
        save_folder_attn = f"./successfull_attn_{eps}/{attack}/Train/Normal"
        print(f'Saving {len(selected_imgs)} images to {save_folder}')
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_attn, exist_ok=True)
        for i, img in enumerate(selected_imgs):
            plt.imsave(os.path.join(save_folder, f"{i}.png"), np.array(img))
            np.save(os.path.join(save_folder_attn, f"{i}.npy"), attns[i])




        pred_i = 1 # 0 --> Normal, 1 --> TB
        print(classes[pred_i])
        print(f'Running {attack} for Epsilon = {eps} on Train TB')
        if attack == 'PDG':
            adv_images_test_normal = apply_pdg(images= images_train_tb, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
        else:
            adv_images_test_normal = apply_fgsm(images= images_train_tb, f_model=f_model, fgsm=fgsm, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=device, eps=eps)

        selected_imgs, attns = get_successful_adv_images(adv_images_test_normal, pred_i)

        save_folder = f"./successful_{eps}/{attack}/Train/TB"
        save_folder_attn = f"./successfull_attn_{eps}/{attack}/Train/TB"
        print(f'Saving {len(selected_imgs)} images to {save_folder}')
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_attn, exist_ok=True)
        for i, img in enumerate(selected_imgs):
            plt.imsave(os.path.join(save_folder, f"{i}.png"), np.array(img))
            np.save(os.path.join(save_folder_attn, f"{i}.npy"), attns[i])

