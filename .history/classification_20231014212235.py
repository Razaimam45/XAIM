import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
# from captum.robust import PGD, FGSM

# from model import vit_base_patch16_224
# from timm.models import vit_base_patch16_224
import sys
# sys.path.append('../')
from saliency import *
from utils import *
from plots import *
import argparse

from saliency import mean_attns_N_images
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomRotation((90,90)), # use with caution (it flips images to side ways)
        transforms.CenterCrop(400),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def get_model(model_path, device):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    model.to(device)
    return model


def get_data(data_path, device):
    data = []
    for img in os.listdir(data_path):
        img_path = os.path.join(data_path, img)
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0).to(device)
        data.append(img)
    return data


def get_data(data_path, device):
    data = []
    img_names = []
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0).to(device)
        data.append(img)
        img_names.append(img_name)
    return data, img_names


if __name__ == "__main__":
    DATA_DIR = "/home/aneeshashmi/xai/data/TB_data/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/aneeshashmi/xai/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth", type=str)
    parser.add_argument("--data_path", default=DATA_DIR, type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="clf_results")
    parser.add_argument("--save_img_path", type=str, default="clf_results_imgs")

    args = parser.parse_args()

    model = get_model(args.model_path, args.device)

    classes = {0: "Normal", 1: "Tuberculosis"}

    for split in ["testing", "training", "validation"]:
    # for split in ["testing"]:
        for selected_class in [0, 1]:
            print(f'Processing {split} {classes[selected_class]}')
            # split = "validation"
            # selected_class = 0
            correctly_clf = []
            correctly_clf_imgs = []
            val_data, val_img_names = get_data(os.path.join(args.data_path, split, classes[selected_class]), args.device)
            for idx, img in tqdm(enumerate(val_data)):
                pred = model(img)
                if pred.argmax() == selected_class:
                    correctly_clf.append(val_img_names[idx])
                    correctly_clf_imgs.append(img.squeeze(0).permute(1,2,0).cpu().numpy())
            # print(correctly_clf_imgs[0])
            print(f'correctly_clf: {len(correctly_clf)}')
            os.makedirs(args.save_path, exist_ok=True)
            # os.makedirs(args.save_img_path, exist_ok=True)
            # for idx, img in enumerate(correctly_clf_imgs):
            #     img_path = os.path.join(args.save_img_path, split, classes[selected_class], correctly_clf[idx])
            #     os.makedirs(os.path.dirname(img_path), exist_ok=True)
            #     plt.imsave(img_path, img, cmap="gray")
            # pd.DataFrame({"correctly_clf": correctly_clf}).to_csv(os.path.join(args.save_path, f"{split}_{classes[selected_class]}_correctly_clf.csv"), index=False)