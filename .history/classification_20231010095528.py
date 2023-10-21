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


def get_model(model_path, device):
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    return model


