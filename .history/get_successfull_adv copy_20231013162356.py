import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import PIL
import foolbox as fb 
import argparse

CLASSES = ["Normal", "Tuberculosis"]
DEVICE = 'cuda'
DATA_PATH = '/home/aneeshashmi/xai/XAIM/clf_results_imgs/testing' #Updated the data path. Picking only correctly classified samples
MODEL_PATH = "/home/aneeshashmi/xai/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth"
BLK = -1
TRANSFORM = transforms.Compose(
[
    transforms.Grayscale(num_output_channels=3),
    # transforms.RandomRotation((90,90)),
    # transforms.CenterCrop(400),
    transforms.Resize((224, 224)),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

def get_model(model_path, device):
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    return model

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

def apply_pdg(f_pgd, f_model, images, device="cuda", eps=0.03, labels=None, target=0, attack_lib='Foolbox'):
    adv_imgs = []
    for input_img in tqdm(images):
        input_img = input_img.float()
        if attack_lib=='Foolbox':
            _, perturbed_image, success = f_pgd(f_model, input_img.to(device), labels.to(device), epsilons=eps)
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
        adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
        adv_imgs.append(adv_img.squeeze(0))
    adv_imgs = torch.stack(adv_imgs)
    return adv_imgs

def load_images(data_path, transform, class_n, device='cuda'):
    images = []
    for img_path in tqdm(os.listdir(os.path.join(data_path, class_n))):
        img = PIL.Image.open(os.path.join(data_path, class_n, img_path))
        img = transform(img).unsqueeze(0).to(device)
        images.append(img)
    return images

def get_success_unsuccess_adv_images (adv_images_list, cls_idx, clean_images, model, image_name_list=None):
    success_imgs = []
    success_imgs_name = []
    unsuccess_imgs = []
    unsuccess_imgs_name = []
    success_imgs_attns = []
    unsuccess_imgs_attns = []
    
    clean_succ_x = []
    clean_unsucc_N_x = []
    clean_succ_x_attns = []
    clean_unsucc_N_x_attns = []

    for i, img in tqdm(enumerate(adv_images_list)):
        attn, pred = get_blk_attn(img.unsqueeze(0).to(DEVICE), BLK, model)
        if pred != cls_idx: # check if pred is not correct
            success_imgs.append(img.permute(1,2,0)) #append to successful adv samples
            success_imgs_attns.append(attn)
            clean_succ_x.append(clean_images[i])
            success_imgs_name.append(image_name_list[i])
            clean_succ_x_attns.append(get_blk_attn(clean_images[i].to(DEVICE), BLK, model))
        else:
            unsuccess_imgs.append(img.permute(1,2,0)) #append to UNsuccessful adv samples
            unsuccess_imgs_attns.append(attn)
            unsuccess_imgs_name.append(image_name_list[i])
            clean_unsucc_N_x.append(clean_images[i])
            clean_unsucc_N_x_attns.append(get_blk_attn(clean_images[i].to(DEVICE), BLK, model))
    return success_imgs, success_imgs_attns, unsuccess_imgs, unsuccess_imgs_attns, \
            clean_succ_x, clean_unsucc_N_x,  clean_succ_x_attns, clean_unsucc_N_x_attns,\
            success_imgs_name, unsuccess_imgs_name

def correctly_classified_images(images_all, cls_idx, model):
    correct_imgs = []
    for i, img in tqdm(enumerate(images_all)):
        _, pred = get_blk_attn(img.to(DEVICE), BLK, model)
        if pred == cls_idx: #checking for correct samples
            correct_imgs.append(img)
    save_dir = f"./correct_data/testing/{CLASSES[cls_idx]}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving {len(correct_imgs)} correctly classified images out of total {len(images_all)} test images")
    for i, _ in tqdm(enumerate(correct_imgs)):
        plt.imsave(os.path.join(save_dir, f"{i}.png"), np.array(correct_imgs[i].squeeze(0).permute(1,2,0).to('cpu')))

def main(DATA_PATH, split="testing"):
    model = get_model(model_path=MODEL_PATH, device=DEVICE)
    # sample_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    # out = model(sample_input)
    # print(out.shape)
    f_model = fb.PyTorchModel(model, bounds=(0,1), device=DEVICE) #Foolbox's PGD # TODO: Check Image min max and compare with bound arg
    f_pgd = fb.attacks.PGD()
    f_model = fb.PyTorchModel(model, bounds=(0,1), device=DEVICE) #Foolbox's PGD
    f_fgsm = fb.attacks.FGSM()

    save_correctly_classified = False
    if save_correctly_classified == True:
        correctly_classified_images(image_test_normal, CLASSES[0], model)
        correctly_classified_images(image_test_tb, CLASSES[1])

    print('loading the correctly classified images')
    normal_testing = [f for f in os.listdir(os.path.join(DATA_PATH, CLASSES[0])) if f.endswith(".jpg") or f.endswith(".png")]
    normal_testing = (normal_testing)[0:]
    tb_testing = [f for f in os.listdir(os.path.join(DATA_PATH, CLASSES[1])) if f.endswith(".jpg") or f.endswith(".png")]
    tb_testing = (tb_testing)[0:]
    # image_test_normal = load_images(DATA_PATH, transform=TRANSFORM, class_n=CLASSES[0], fname_list=normal_testing)
    # image_test_tb = load_images(DATA_PATH, transform=TRANSFORM, class_n=CLASSES[1], fname_list=tb_testing)
    image_test_normal = load_images(DATA_PATH, transform=TRANSFORM, class_n=CLASSES[0])
    image_test_tb = load_images(DATA_PATH, transform=TRANSFORM, class_n=CLASSES[1])
    print('loading done')

    for eps in [0.03, 0.06, 0.01]:
        for attack in ['PDG', 'FGSM']:
            class_name = 'Normal'      
            pred_i = 0 # 0 --> Normal, 1 --> TB
            print(CLASSES[pred_i])
            print(f'Running {attack} for Epsilon = {eps} on Test normal')
            if attack == 'PDG':
                adv_images_test_normal = apply_pdg(images = image_test_normal, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
            else:
                adv_images_test_normal = apply_fgsm(images=  image_test_normal, f_model=f_model, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=DEVICE, eps=eps)
            normal_testing_filenames = [i[:-4] for i in normal_testing]
            succ_imgs, succ_attns, unsucc_imgs, unsucc_attns, clean_succ_x, clean_unsucc_N_x, clean_succ_x_attns, clean_unsucc_N_x_attns, success_imgs_name, unsuccess_imgs_name = get_success_unsuccess_adv_images(adv_images_test_normal, pred_i, image_test_normal, normal_testing_filenames)
            
            save_dir_succ_x = f".{split}/data4/{attack}_{eps}/{class_name}/Succ/Succ_x/"
            save_dir_succ_clean_x = f".{split}/data4/{attack}_{eps}/{class_name}/Succ/Clean_x/"
            save_dir_unsucc_N_x = f".{split}/data4/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x/"
            save_dir_unsucc_clean_N_x = f".{split}/data4/{attack}_{eps}/{class_name}/Unsucc/Clean_x/"
            
            save_dir_succ_x_attn = f"./data4/{attack}_{eps}/{class_name}/Succ/Succ_x_attn/"
            save_dir_succ_clean_x_attn = f"./data4/{attack}_{eps}/{class_name}/Succ/Clean_x_attn/"
            save_dir_unsucc_N_x_attn = f"./data4/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x_attn/"
            save_dir_unsucc_clean_N_x_attn = f"./data4/{attack}_{eps}/{class_name}/Unsucc/Clean_x_attn/"
            
            os.makedirs(save_dir_succ_x, exist_ok=True)
            os.makedirs(save_dir_succ_clean_x, exist_ok=True)
            os.makedirs(save_dir_unsucc_N_x, exist_ok=True)
            os.makedirs(save_dir_unsucc_clean_N_x, exist_ok=True)  
            
            os.makedirs(save_dir_succ_x_attn, exist_ok=True)
            os.makedirs(save_dir_succ_clean_x_attn, exist_ok=True)
            os.makedirs(save_dir_unsucc_N_x_attn, exist_ok=True)
            os.makedirs(save_dir_unsucc_clean_N_x_attn, exist_ok=True)  
                    
            for i, _ in tqdm(enumerate(succ_imgs)):
                plt.imsave(os.path.join(save_dir_succ_x, f"{success_imgs_name[i]}.png"), np.array(succ_imgs[i]))
                np.save(os.path.join(save_dir_succ_x_attn, f"{success_imgs_name[i]}.npy"), succ_attns[i])
                plt.imsave(os.path.join(save_dir_succ_clean_x, f"{success_imgs_name[i]}.png"), np.array(clean_succ_x[i].squeeze(0).permute(1,2,0).to('cpu')))
                np.save(os.path.join(save_dir_succ_clean_x_attn, f"{success_imgs_name[i]}.npy"), clean_succ_x_attns[i])
                
            for i, _ in tqdm(enumerate(unsucc_imgs)):
                plt.imsave(os.path.join(save_dir_unsucc_N_x, f"{unsuccess_imgs_name[i]}.png"), np.array(unsucc_imgs[i]))
                np.save(os.path.join(save_dir_unsucc_N_x_attn, f"{unsuccess_imgs_name[i]}.npy"), unsucc_attns[i])
                plt.imsave(os.path.join(save_dir_unsucc_clean_N_x, f"{unsuccess_imgs_name[i]}.png"), np.array(clean_unsucc_N_x[i].squeeze(0).permute(1,2,0).to('cpu')))
                np.save(os.path.join(save_dir_unsucc_clean_N_x_attn, f"{unsuccess_imgs_name[i]}.npy"), clean_unsucc_N_x_attns[i])

            print("----------------------------------------------")
            print("----------------------------------------------")
            
            # for TB class
            class_name = 'TB'      
            pred_i = 1 # 0 --> Normal, 1 --> TB
            print(CLASSES[pred_i])
            print(f'Running {attack} for Epsilon = {eps} on Test TB')
            if attack == 'PDG':
                adv_images_test_tb = apply_pdg(images=  image_test_tb, f_model=f_model, f_pgd=f_pgd, labels=torch.tensor([pred_i]), eps = eps)
            else:
                adv_images_test_tb = apply_fgsm(images=  image_test_tb, f_model=f_model, f_fgsm = f_fgsm, labels=torch.tensor([pred_i]), device=DEVICE, eps=eps)
            tb_testing_filenames = [i[:-4] for i in tb_testing]
            succ_imgs, succ_attns, unsucc_imgs, unsucc_attns, clean_succ_x, clean_unsucc_N_x, clean_succ_x_attns, clean_unsucc_N_x_attns, success_imgs_name, unsuccess_imgs_name = get_success_unsuccess_adv_images(adv_images_test_tb, pred_i, image_test_tb, tb_testing_filenames)
            
            save_dir_succ_x = f"./data4/{attack}_{eps}/{class_name}/Succ/Succ_x/"
            save_dir_succ_clean_x = f"./data4/{attack}_{eps}/{class_name}/Succ/Clean_x/"
            save_dir_unsucc_N_x = f"./data4/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x/"
            save_dir_unsucc_clean_N_x = f"./data4/{attack}_{eps}/{class_name}/Unsucc/Clean_x/"
            
            save_dir_succ_x_attn = f"./data4/{attack}_{eps}/{class_name}/Succ/Succ_x_attn/"
            save_dir_succ_clean_x_attn = f"./data4/{attack}_{eps}/{class_name}/Succ/Clean_x_attn/"
            save_dir_unsucc_N_x_attn = f"./data4/{attack}_{eps}/{class_name}/Unsucc/Unsucc_x_attn/"
            save_dir_unsucc_clean_N_x_attn = f"./data4/{attack}_{eps}/{class_name}/Unsucc/Clean_x_attn/"
            
            os.makedirs(save_dir_succ_x, exist_ok=True)
            os.makedirs(save_dir_succ_clean_x, exist_ok=True)
            os.makedirs(save_dir_unsucc_N_x, exist_ok=True)
            os.makedirs(save_dir_unsucc_clean_N_x, exist_ok=True)  
            
            os.makedirs(save_dir_succ_x_attn, exist_ok=True)
            os.makedirs(save_dir_succ_clean_x_attn, exist_ok=True)
            os.makedirs(save_dir_unsucc_N_x_attn, exist_ok=True)
            os.makedirs(save_dir_unsucc_clean_N_x_attn, exist_ok=True)  
                    
            for i, _ in tqdm(enumerate(succ_imgs)):
                plt.imsave(os.path.join(save_dir_succ_x, f"{success_imgs_name[i]}.png"), np.array(succ_imgs[i]))
                np.save(os.path.join(save_dir_succ_x_attn, f"{success_imgs_name[i]}.npy"), succ_attns[i])
                plt.imsave(os.path.join(save_dir_succ_clean_x, f"{success_imgs_name[i]}.png"), np.array(clean_succ_x[i].squeeze(0).permute(1,2,0).to('cpu')))
                np.save(os.path.join(save_dir_succ_clean_x_attn, f"{success_imgs_name[i]}.npy"), clean_succ_x_attns[i])
                
            for i, _ in tqdm(enumerate(unsucc_imgs)):
                plt.imsave(os.path.join(save_dir_unsucc_N_x, f"{unsuccess_imgs_name[i]}.png"), np.array(unsucc_imgs[i]))
                np.save(os.path.join(save_dir_unsucc_N_x_attn, f"{unsuccess_imgs_name[i]}.npy"), unsucc_attns[i])
                plt.imsave(os.path.join(save_dir_unsucc_clean_N_x, f"{unsuccess_imgs_name[i]}.png"), np.array(clean_unsucc_N_x[i].squeeze(0).permute(1,2,0).to('cpu')))
                np.save(os.path.join(save_dir_unsucc_clean_N_x_attn, f"{unsuccess_imgs_name[i]}.npy"), clean_unsucc_N_x_attns[i])

if __name__ == "__main__":
    for split in ['testing', 'training', 'validation']:
        DATA_PATH = f'/home/aneeshashmi/xai/XAIM/clf_results_imgs/{split}' #Updated the data path. Picking only correctly classified samples
        main(DATA_PATH=DATA_PATH, split=split)