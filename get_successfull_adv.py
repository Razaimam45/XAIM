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
model_path = "/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth"

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
    mean_attention = np.mean(attentions, 0)

    return mean_attention, torch.argmax(out).item()

def folder_sort(image_files):
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
    return image_files

# %%
classes = ["Normal", "Tuberculosis"]
device = 'cuda'
data_path = '/home/raza.imam/Documents/HC701B/Project/data/TB_data'

# %%
model = get_model(model_path=model_path, device=device)


normal_testing = [f for f in os.listdir(os.path.join(data_path, 'testing', classes[0])) if f.endswith(".jpg") or f.endswith(".png")]
normal_testing = (normal_testing)[0:]
tb_testing = [f for f in os.listdir(os.path.join(data_path, 'testing', classes[1])) if f.endswith(".jpg") or f.endswith(".png")]
tb_testing = (tb_testing)[0:]

blk = -1

# %%
# from captum.robust import PGD, FGSM
import foolbox as fb


# %%
def apply_pdg(f_pgd, f_model, images, device="cuda", eps=0.03, labels=None, target=0, attack_lib='Foolbox'):
    # if pgd is None:
    #     pgd = PGD(model, lower_bound=0, upper_bound=1)

    # labels = torch.tensor([1])
    adv_imgs = []
    for input_img in tqdm(images):
        input_img = input_img.float()
        if attack_lib=='Foolbox':
            _, perturbed_image, success = f_pgd(f_model, input_img.to(device), labels.to(device), epsilons=eps)
        # else:
        #     perturbed_image = pgd.perturb(input_img.to(device), radius=radius, step_size=eps, step_num=step_num, target=target) #step_size = epsilon in PGD case
        adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
        
        adv_imgs.append(adv_img.squeeze(0))
    adv_imgs = torch.stack(adv_imgs)
    return adv_imgs


def apply_fgsm(images, device="cuda", eps=0.03, attack_lib="Foolbox", labels = None, f_fgsm = None, f_model = None):    
    adv_imgs = []
    for input_img in tqdm(images):
        input_img = input_img.float()
        if attack_lib=='Foolbox':
            _, perturbed_image, success = f_fgsm(f_model, input_img.to(device), labels.to(device), epsilons=eps)
        # else:
        #     perturbed_image = fgsm.perturb(input_img.to(device), epsilon=eps, target=0) 
        adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
        adv_imgs.append(adv_img.squeeze(0))
    adv_imgs = torch.stack(adv_imgs)
    return adv_imgs


print('loading the images')
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
print('loading done')

def get_success_unsuccess_adv_images (adv_images_list, cls_idx, clean_images):
    success_imgs = []
    unsuccess_imgs = []
    success_imgs_attns = []
    unsuccess_imgs_attns = []
    
    clean_succ_x = []
    clean_unsucc_N_x = []
    clean_succ_x_attns = []
    clean_unsucc_N_x_attns = []

    for i, img in tqdm(enumerate(adv_images_list)):
        attn, pred = get_blk_attn(img.unsqueeze(0).to(device), blk, model)
        if pred != cls_idx: # check if pred is not correct
            success_imgs.append(img.permute(1,2,0)) #append to successful adv samples
            success_imgs_attns.append(attn)
            clean_succ_x.append(clean_images[i])
            clean_succ_x_attns.append(get_blk_attn(clean_images[i].to(device), blk, model))
            
        else:
            unsuccess_imgs.append(img.permute(1,2,0)) #append to UNsuccessful adv samples
            unsuccess_imgs_attns.append(attn)
            clean_unsucc_N_x.append(clean_images[i])
            clean_unsucc_N_x_attns.append(get_blk_attn(clean_images[i].to(device), blk, model))
            
            
    return success_imgs, success_imgs_attns, unsuccess_imgs, unsuccess_imgs_attns, clean_succ_x, clean_unsucc_N_x,  clean_succ_x_attns, clean_unsucc_N_x_attns 

# -------------------------------------- # 



# #Saving clean test Normal
# save_folder_cleans = f"./data2/clean/Test/Normal"
# print(f'Saving {len(image_test_normal)} cleans to {save_folder_cleans}')
# os.makedirs(save_folder_cleans, exist_ok=True)
# for i, img in tqdm(enumerate(image_test_normal)):
#     img = img.squeeze(0).squeeze(0).permute(1, 2 ,0).to('cpu')
#     plt.imsave(os.path.join(save_folder_cleans, f"{i}.png"), np.array(img))
    
# #Saving clean test TB
# save_folder_cleans = f"./data2/clean/Test/TB"
# print(f'Saving {len(image_test_tb)} cleans to {save_folder_cleans}')
# os.makedirs(save_folder_cleans, exist_ok=True)
# for i, img in tqdm(enumerate(image_test_tb)):
#     img = img.squeeze(0).squeeze(0).permute(1, 2, 0).to('cpu')
#     plt.imsave(os.path.join(save_folder_cleans, f"{i}.png"), np.array(img))


# %%
f_model = fb.PyTorchModel(model, bounds=(0,1), device=device) #Foolbox's PGD
f_pgd = fb.attacks.PGD()

f_model = fb.PyTorchModel(model, bounds=(0,1), device=device) #Foolbox's PGD
f_fgsm = fb.attacks.FGSM()

for eps in [0.03, 0.06, 0.01]:
    for attack in ['PDG', 'FGSM']:
        class_name = 'Normal'      
        pred_i = 0 # 0 --> Normal, 1 --> TB
        print(classes[pred_i])
        print(f'Running {attack} for Epsilon = {eps} on Test normal')
        if attack == 'PDG':
            adv_images_test_normal = apply_pdg(images=  image_test_normal, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
        else:
            adv_images_test_normal = apply_fgsm(images=  image_test_normal, f_model=f_model, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=device, eps=eps)
        succ_imgs, succ_attns, unsucc_imgs, unsucc_attns, clean_succ_x, clean_unsucc_N_x, clean_succ_x_attns, clean_unsucc_N_x_attns = get_success_unsuccess_adv_images(adv_images_test_normal, pred_i, image_test_normal)
        
        save_dir_succ_x = f"./data3/{attack}_{eps}/{class_name}/Succ/Succ_x/"
        save_dir_succ_clean_x = f"./data3/{attack}_{eps}/{class_name}/Succ/Clean_x/"
        save_dir_unsucc_N_x = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x/"
        save_dir_unsucc_clean_N_x = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Clean_x/"
        
        save_dir_succ_x_attn = f"./data3/{attack}_{eps}/{class_name}/Succ/Succ_x_attn/"
        save_dir_succ_clean_x_attn = f"./data3/{attack}_{eps}/{class_name}/Succ/Clean_x_attn/"
        save_dir_unsucc_N_x_attn = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x_attn/"
        save_dir_unsucc_clean_N_x_attn = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Clean_x_attn/"
        
        os.makedirs(save_dir_succ_x, exist_ok=True)
        os.makedirs(save_dir_succ_clean_x, exist_ok=True)
        os.makedirs(save_dir_unsucc_N_x, exist_ok=True)
        os.makedirs(save_dir_unsucc_clean_N_x, exist_ok=True)  
        
        os.makedirs(save_dir_succ_x_attn, exist_ok=True)
        os.makedirs(save_dir_succ_clean_x_attn, exist_ok=True)
        os.makedirs(save_dir_unsucc_N_x_attn, exist_ok=True)
        os.makedirs(save_dir_unsucc_clean_N_x_attn, exist_ok=True)  
                
        for i, _ in tqdm(enumerate(succ_imgs)):
            plt.imsave(os.path.join(save_dir_succ_x, f"{i}.png"), np.array(succ_imgs[i]))
            np.save(os.path.join(save_dir_succ_x_attn, f"{i}.npy"), succ_attns[i])
            plt.imsave(os.path.join(save_dir_succ_clean_x, f"{i}.png"), np.array(clean_succ_x[i].squeeze(0).permute(1,2,0).to('cpu')))
            np.save(os.path.join(save_dir_succ_clean_x_attn, f"{i}.npy"), clean_succ_x_attns[i])
            
        for i, _ in tqdm(enumerate(unsucc_imgs)):
            plt.imsave(os.path.join(save_dir_unsucc_N_x, f"{i}.png"), np.array(unsucc_imgs[i]))
            np.save(os.path.join(save_dir_unsucc_N_x_attn, f"{i}.npy"), unsucc_attns[i])
            plt.imsave(os.path.join(save_dir_unsucc_clean_N_x, f"{i}.png"), np.array(clean_unsucc_N_x[i].squeeze(0).permute(1,2,0).to('cpu')))
            np.save(os.path.join(save_dir_unsucc_clean_N_x_attn, f"{i}.npy"), clean_unsucc_N_x_attns[i])

        print("----------------------------------------------")
        print("----------------------------------------------")
        
        # for TB class
        class_name = 'TB'      
        pred_i = 1 # 0 --> Normal, 1 --> TB
        print(classes[pred_i])
        print(f'Running {attack} for Epsilon = {eps} on Test TB')
        if attack == 'PDG':
            adv_images_test_tb = apply_pdg(images=  image_test_tb, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
        else:
            adv_images_test_tb = apply_fgsm(images=  image_test_tb, f_model=f_model, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=device, eps=eps)
        succ_imgs, succ_attns, unsucc_imgs, unsucc_attns, clean_succ_x, clean_unsucc_N_x, clean_succ_x_attns, clean_unsucc_N_x_attns = get_success_unsuccess_adv_images(adv_images_test_tb, pred_i, image_test_tb)
        
        save_dir_succ_x = f"./data3/{attack}_{eps}/{class_name}/Succ/Succ_x/"
        save_dir_succ_clean_x = f"./data3/{attack}_{eps}/{class_name}/Succ/Clean_x/"
        save_dir_unsucc_N_x = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x/"
        save_dir_unsucc_clean_N_x = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Clean_x/"
        
        save_dir_succ_x_attn = f"./data3/{attack}_{eps}/{class_name}/Succ/Succ_x_attn/"
        save_dir_succ_clean_x_attn = f"./data3/{attack}_{eps}/{class_name}/Succ/Clean_x_attn/"
        save_dir_unsucc_N_x_attn = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x_attn/"
        save_dir_unsucc_clean_N_x_attn = f"./data3/{attack}_{eps}/{class_name}/Unsucc/Clean_x_attn/"
        
        os.makedirs(save_dir_succ_x, exist_ok=True)
        os.makedirs(save_dir_succ_clean_x, exist_ok=True)
        os.makedirs(save_dir_unsucc_N_x, exist_ok=True)
        os.makedirs(save_dir_unsucc_clean_N_x, exist_ok=True)  
        
        os.makedirs(save_dir_succ_x_attn, exist_ok=True)
        os.makedirs(save_dir_succ_clean_x_attn, exist_ok=True)
        os.makedirs(save_dir_unsucc_N_x_attn, exist_ok=True)
        os.makedirs(save_dir_unsucc_clean_N_x_attn, exist_ok=True)  
                
        for i, _ in tqdm(enumerate(succ_imgs)):
            plt.imsave(os.path.join(save_dir_succ_x, f"{i}.png"), np.array(succ_imgs[i]))
            np.save(os.path.join(save_dir_succ_x_attn, f"{i}.npy"), succ_attns[i])
            plt.imsave(os.path.join(save_dir_succ_clean_x, f"{i}.png"), np.array(clean_succ_x[i].squeeze(0).permute(1,2,0).to('cpu')))
            np.save(os.path.join(save_dir_succ_clean_x_attn, f"{i}.npy"), clean_succ_x_attns[i])
            
        for i, _ in tqdm(enumerate(unsucc_imgs)):
            plt.imsave(os.path.join(save_dir_unsucc_N_x, f"{i}.png"), np.array(unsucc_imgs[i]))
            np.save(os.path.join(save_dir_unsucc_N_x_attn, f"{i}.npy"), unsucc_attns[i])
            plt.imsave(os.path.join(save_dir_unsucc_clean_N_x, f"{i}.png"), np.array(clean_unsucc_N_x[i].squeeze(0).permute(1,2,0).to('cpu')))
            np.save(os.path.join(save_dir_unsucc_clean_N_x_attn, f"{i}.npy"), clean_unsucc_N_x_attns[i])
