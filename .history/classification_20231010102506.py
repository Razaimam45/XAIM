import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from captum.robust import PGD, FGSM

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
import torchvision


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomRotation((90,90)),
        transforms.CenterCrop(400),
        transforms.Resize((224, 224)),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]
)


def get_model(model_path, device):
    model = torch.load(model_path)
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
    DATA_DIR = "/home/aneeshashmi/xai/data/TB_data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/aneeshashmi/xai/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth", type=str)
    parser.add_argument("--data_path", default=DATA_DIR, type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="clf_results")

    args = parser.parse_args()

    model = get_model(args.model_path, args.device)

    classes = {0: "Normal", 1: "Tubercolosis"}


    split = "validation"
    correctly_clf = []
    selected_class = 0
    val_data, val_img_names = get_data(os.path.join(args.data_path, split, classes[selected_class]), args.device)
    for idx, img in tqdm(enumerate(val_data)):
        pred = model(img)
        if pred.argmax() == selected_class:
            print("True")
            correctly_clf.append(val_img_names[idx])
        break
    os.makedirs(args.save_path, exist_ok=True)
    pd.DataFrame({"correctly_clf": correctly_clf}).to_csv(os.path.join(args.save_path, f"{split}_{classes[selected_class]}_correctly_clf.csv"), index=False)



# train_dataset = torchvision.datasets.ImageFolder(
#     root=os.path.join(DATA_DIR, "training"), transform=transform
# )

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=1, shuffle=True, num_workers=2
# )

# val_dataset = torchvision.datasets.ImageFolder(
#     root=os.path.join(DATA_DIR, "validation"), transform=transform
# )

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=1, shuffle=True, num_workers=2
# )

# test_dataset = torchvision.datasets.ImageFolder(