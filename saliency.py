
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from captum.robust import PGD
from captum.attr import visualization as viz
from captum.attr import Saliency
import random
from plots import plot_sal_hist_kde

def get_attn(input_img, block, head, classifier):   
    output, attn = classifier(input_img, return_attn=True)
    normal_attn_map = attn[block][0][head][:-1,-1].reshape(14, -1).detach().cpu().numpy()
    return normal_attn_map

def mean_attn_heads(block, N_heads, input_img, classifier): #N_heads=first N heads of a block
    # returns mean attention of N_heads for Input image input_image
    attentions = []
    for head in range(N_heads):
        # print(f"For Block {block} head {head}")
        attn = get_attn(input_img.unsqueeze(0), block, head, classifier)
        # saliency = (saliency -saliency.min()) / (saliency.max() -saliency.min()) # Normalizing Normal ---
        attentions.append(attn)

    mean_attn_of_N_heads = np.mean(attentions, axis=0)
    return mean_attn_of_N_heads

def mean_saliency_generate(N_images, image_folder, block, head, classifier, N_random=True, N_heads=1):
    # image_folder = '/home/raza.imam/Documents/HC701B/Project/data/TB_data/training/Normal'

    transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation (20),
                transforms.Resize((224,224)),
                transforms.RandomAffine(degrees =0,translate=(0.1,0.1)),
                transforms.ToTensor()
    ])

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    N, rand = N_images, N_random
    if rand==False:
        image_files = image_files[:N]  #first N images
    else:
        image_files = random.sample(image_files, N) #random N images

    images = []
    for f in image_files:
        image_path = os.path.join(image_folder, f)
        image = Image.open(image_path)
        image = transform(image)
        image = (image -image.min()) / (image.max() -image.min())
        images.append(image)

    images_tensor = torch.stack(images)

    saliencies_normal = []
    for input_img in images_tensor:
        # print(input_img.min(), input_img.max())
        if N_heads==1: # For 1 particular head
            saliency = get_attn(input_img.unsqueeze(0), block, head, classifier)
        else: # For N heads of a block
            saliency = mean_attn_heads(block, N_heads, input_img, classifier)
        # saliency = (saliency -saliency.min()) / (saliency.max() -saliency.min()) # Normalizing Normal ---
        saliencies_normal.append(saliency)

    mean_saliency_normal = np.mean(saliencies_normal, axis=0)
    print("mean_saliency_normal.shape:", mean_saliency_normal.shape)

    pgd = PGD(classifier, lower_bound=0, upper_bound=1)
    adv_images = []
    for input_img in images_tensor:
        input_img = input_img.unsqueeze(0).float()
        perturbed_image = pgd.perturb(input_img, radius=0.13, step_size=0.02, step_num=7, target=0) 
        adv_sample = torch.tensor((perturbed_image.cpu().data.numpy()))
        adv_images.append(adv_sample.squeeze(0))

    adv_images_tensor = torch.stack(adv_images)

    saliencies_adv = []
    for adv_img in adv_images_tensor:
        if N_heads==1: # For 1 particular head
            saliency = get_attn(adv_img.unsqueeze(0), block, head, classifier)
        else: # For N heads of a block
            saliency = mean_attn_heads(block, N_heads, adv_img, classifier)
        # saliency = (saliency -saliency.min()) / (saliency.max() -saliency.min()) # Normalizing Adv ---
        saliencies_adv.append(saliency)

    mean_saliency_adv = np.mean(saliencies_adv, axis=0)
    print("mean_saliency_adv.shape:", mean_saliency_adv.shape)    

    saliency_diff = mean_saliency_adv - mean_saliency_normal
    print("saliency_diff.shape:", saliency_diff.shape)    


    return mean_saliency_normal, mean_saliency_adv, saliency_diff, saliencies_normal, saliencies_adv


def test_img_saliency(image_folder, block, head, classifier, plot=False, rand=True):
    if rand==True:
        f_0 = random.choice(os.listdir(image_folder))
    else:
        f_0 = os.listdir(image_folder)[0]
    image_path = os.path.join(image_folder, f_0)
    transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation (20),
                transforms.Resize((224,224)),
                transforms.RandomAffine(degrees =0,translate=(0.1,0.1)),
                transforms.ToTensor()
    ])
    image_0 = transform(Image.open(image_path))
    image_0 = (image_0 -image_0.min()) / (image_0.max() -image_0.min())
    image_tensor_0 = image_0.unsqueeze(0)
    # print(image_tensor_0.min(), image_tensor_0.max())

    saliency_0 = get_attn(image_tensor_0, block, head, classifier)
    saliency_normal_0 = saliency_0
    # saliency_normal_0 = (saliency_0 -saliency_0.min()) / (saliency_0.max() -saliency_0.min()) # Normalizing ...

    pgd_0 = PGD(classifier, lower_bound=0, upper_bound=1)
    input_img_0 = image_tensor_0.float()
    perturbed_image_0 = pgd_0.perturb(input_img_0, radius=0.13, step_size=0.02, step_num=7, target=0)
    adv_sample_0 = perturbed_image_0.cpu().data.numpy()
    adv_image_tensor_0 = torch.tensor(adv_sample_0)

    adv_saliency_0 = get_attn(adv_image_tensor_0, block, head, classifier)
    saliency_adv_0 = adv_saliency_0

    saliency_diff = saliency_adv_0 - saliency_normal_0
    
    if plot==True:
        plot_sal_hist_kde(saliency_normal_0, saliency_adv_0, saliency_diff, 1, kde=False, pca=False)
    
    return f_0, saliency_normal_0, saliency_adv_0, saliency_diff