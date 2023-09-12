
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
from plots import *
import sys
sys.path.insert(1, '/home/raza.imam/Documents/HC701B/Project/pytorch-grad-cam')
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.hirescam import HiResCAM
from pytorch_grad_cam.grad_cam_elementwise import GradCAMElementWise
from pytorch_grad_cam.ablation_layer import AblationLayer, AblationLayerVit, AblationLayerFasterRCNN
from pytorch_grad_cam.ablation_cam import AblationCAM
from pytorch_grad_cam.xgrad_cam import XGradCAM
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.score_cam import ScoreCAM
from pytorch_grad_cam.layer_cam import LayerCAM
from pytorch_grad_cam.eigen_cam import EigenCAM
from pytorch_grad_cam.eigen_grad_cam import EigenGradCAM
from pytorch_grad_cam.random_cam import RandomCAM
from pytorch_grad_cam.fullgrad_cam import FullGrad
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.feature_factorization.deep_feature_factorization import DeepFeatureFactorization, run_dff_on_image
import pytorch_grad_cam.utils.model_targets
import pytorch_grad_cam.utils.reshape_transforms
import pytorch_grad_cam.metrics.cam_mult_image
import pytorch_grad_cam.metrics.road


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_pyt_saliency(input_img, block, head, classifier):
    target_layers = [classifier.blocks[block].attn]

    method = "gradcam"
    # Instantiate the selected method
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        # "fullgrad": FullGrad
    }
    if method not in methods:
        raise Exception(f"Method {method} not implemented")
    if method == "ablationcam":
        cam = methods[method](model=classifier,
                            target_layers=target_layers,
                            use_cuda=True,
                            reshape_transform=reshape_transform,
                            ablation_layer=AblationLayerVit())
    else:
        cam = methods[method](model=classifier,
                            target_layers=target_layers,
                            use_cuda=True,
                            reshape_transform=reshape_transform)
        
    grayscale_cam = cam(input_tensor=input_img,
                        targets=None,
                        eigen_smooth=False,
                        aug_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam
        

def get_attn(input_img, block, head, classifier):   
    output, attn = classifier(input_img, return_attn=True)
    normal_attn_map = attn[block][0][head][:-1,-1].reshape(14, -1).detach().cpu().numpy()
    normal_attn_map = (normal_attn_map -normal_attn_map.min()) / (normal_attn_map.max() -normal_attn_map.min())
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
    mean_attn_of_N_heads = (mean_attn_of_N_heads - mean_attn_of_N_heads.min())/(mean_attn_of_N_heads.max()-mean_attn_of_N_heads.min())

    return mean_attn_of_N_heads

def multiply_attn_heads(block, N_heads, input_img, classifier): #N_heads=first N heads of a block
    # returns mean attention of N_heads for Input image input_image
    attentions = []
    for head in range(N_heads):
        # print(f"For Block {block} head {head}")
        attn = get_attn(input_img.unsqueeze(0), block, head, classifier)
        # saliency = (saliency -saliency.min()) / (saliency.max() -saliency.min()) # Normalizing Normal ---
        attentions.append(attn)

    prod_attn = np.ones_like(attentions[0])
    for attn in attentions:
        prod_attn *= attn
    prod_attn = (prod_attn - prod_attn.min())/(prod_attn.max()-prod_attn.min())
    return prod_attn

def mean_saliency_generate(N_images, image_folder, block, head, classifier, N_random=True, N_heads=1, get_sal = get_attn, agg_attn_heads=mean_attn_heads):
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
            saliency = get_sal(input_img.unsqueeze(0), block, head, classifier)
        else: # For N heads of a block
            saliency = agg_attn_heads(block, N_heads, input_img, classifier)
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
            saliency = get_sal(adv_img.unsqueeze(0), block, head, classifier)
        else: # For N heads of a block
            saliency = agg_attn_heads(block, N_heads, adv_img, classifier)
        # saliency = (saliency -saliency.min()) / (saliency.max() -saliency.min()) # Normalizing Adv ---
        saliencies_adv.append(saliency)

    mean_saliency_adv = np.mean(saliencies_adv, axis=0)
    print("mean_saliency_adv.shape:", mean_saliency_adv.shape)    

    saliency_diff = mean_saliency_adv - mean_saliency_normal
    print("saliency_diff.shape:", saliency_diff.shape)    

    print(mean_saliency_normal.shape)
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




#DEBUG: Phase 2 ------------
from torch import nn

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

    return mean_attention #REVIEW:Return mean of 12 head attentions

    
def mean_attns_N_images(image_folder, N_images, block, model, N_random=True, device="cuda", N_heads=12):
    # image_folder = '/home/raza.imam/Documents/HC701B/Project/data/TB_data/training/Normal'

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

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]    
    if N_random==False:
        image_files = image_files[:N_images]  #first N images
    else:
        image_files = random.sample(image_files, N_images) #random N images

    #REVIEW:creating list of N clean images
    images = []
    for f in image_files:
        image_path = os.path.join(image_folder, f)
        image = Image.open(image_path)
        image = transform(image)
        # image = image.unsqueeze(0)        
        images.append(image)
    images_tensor = torch.stack(images)
    print(images_tensor.shape)
    
    #REVIEW:Calculating Attention for each clean image
    attentions_clean = []
    for input_img in images_tensor:
        input_img = input_img.unsqueeze(0)
        attn = get_blk_attn(input_img.to(device), block, model)
        attentions_clean.append(attn)
    mean_attns_cln = np.mean(attentions_clean, axis=0)
    print("mean_attns_cln.shape:", mean_attns_cln.shape)
    
    #REVIEW:Creating list of N adv images
    pgd = PGD(model, lower_bound=0, upper_bound=1)
    adv_images = []
    for input_img in images_tensor:
        input_img = input_img.unsqueeze(0).float()
        perturbed_image = pgd.perturb(input_img.to(device), radius=0.13, step_size=0.02, step_num=7, target=0) 
        adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
        adv_images.append(adv_img.squeeze(0))
    adv_images_tensor = torch.stack(adv_images)

    #REVIEW:Calculating Attention for each adv image
    attentions_adv = []
    for adv_img in adv_images_tensor:
        adv_img = adv_img.unsqueeze(0)
        attn = get_blk_attn(adv_img.to(device), block, model)
        attentions_adv.append(attn)
    mean_attns_adv = np.mean(attentions_adv, axis=0)
    print("mean_attns_adv.shape:", mean_attns_adv.shape)    

    # Calculating difference of mean_attns_cln and mean_attns_adv
    mean_attns_diff = mean_attns_adv - mean_attns_cln
    print("mean_attns_diff.shape:", mean_attns_diff.shape)    

    return mean_attns_cln, mean_attns_adv, mean_attns_diff, attentions_clean, attentions_adv


def test_img_attn(image_folder, block, model, plot=False, rand=True, random_state=None):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]    
    if rand==False:
        f_0 = image_files[:1][0]  #first N images
    else:
        random.seed(random_state)
        f_0 = random.sample(image_files, 1)[0] #random N images

    transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation((90,90)),
        transforms.CenterCrop(400),
        transforms.Resize((224, 224)),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    # Getting block attention for clean version
    image_path = os.path.join(image_folder, f_0)
    img = transform(Image.open(image_path))
    img = img.unsqueeze(0)
    attn_cln = get_blk_attn(img.cuda(), block, model)

    # Generating adv sample and getting block attention
    pgd = PGD(model, lower_bound=0, upper_bound=1)
    img = img.float()
    perturbed_image = pgd.perturb(img.cuda(), radius=0.13, step_size=0.02, step_num=7, target=0) 
    adv_img = torch.tensor((perturbed_image.cpu().data.numpy()))
    attn_adv = get_blk_attn(adv_img.cuda(), block, model)

    att_diff = attn_adv - attn_cln
    
    if plot==True:
        plot_statistics(attn_cln, attn_adv, N_images=1, hist=True, kde=False, pca=False)
    
    return img, attn_cln, attn_adv, att_diff

