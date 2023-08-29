import os
import numpy as np
from PIL import Image
import seaborn as sns
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def sum_metric(saliency_test_img, mean_saliency_normal, mean_saliency_adv):
    del_N = (mean_saliency_normal-saliency_test_img)
    del_A = (mean_saliency_adv-saliency_test_img)
    # print(del_N.shape, del_A.shape)

    sum_N = np.sum(del_N)
    sum_A = np.sum(del_A)

    return sum_N, sum_A

def cosine_similarity_metric(saliency_test_img, mean_saliency_normal, mean_saliency_adv):
    cosine_N = np.dot(saliency_test_img, mean_saliency_normal)/(norm(mean_saliency_normal)*norm(saliency_test_img))
    cosine_A = np.dot(saliency_test_img, mean_saliency_adv)/(norm(mean_saliency_adv)*norm(saliency_test_img))
    # print(del_N.shape, del_A.shape)

    sum_N = np.sum(cosine_N)
    sum_A = np.sum(cosine_A)

    return sum_N, sum_A

def l2_distance_metric(saliency_test_img, mean_saliency_normal, mean_saliency_adv):
    l2_N = np.linalg.norm(mean_saliency_normal - saliency_test_img)
    l2_A = np.linalg.norm(mean_saliency_adv - saliency_test_img)

    return l2_N, l2_A

def frobenius_norm_metric(saliency_test_img, mean_saliency_normal, mean_saliency_adv):
    frobenius_N = np.linalg.norm(saliency_test_img - mean_saliency_normal, ord='fro')
    frobenius_A = np.linalg.norm(saliency_test_img - mean_saliency_adv, ord='fro')

    return frobenius_N, frobenius_A

def kl_divergence_metric(saliency_test_img, mean_saliency_normal, mean_saliency_adv):
    epsilon = 1e-8  # Small constant to avoid division by zero

    p = saliency_test_img.flatten() + epsilon
    q_normal = mean_saliency_normal.flatten() + epsilon
    q_adv = mean_saliency_adv.flatten() + epsilon

    kl_divergence_N = np.sum(p * np.log(p / q_normal))
    kl_divergence_A = np.sum(p * np.log(p / q_adv))

    return kl_divergence_N, kl_divergence_A

from sklearn.metrics.pairwise import cosine_similarity
def cosine_similarity_1D(saliency_test_img, mean_saliency_normal, mean_saliency_adv):
    
    cosine_sim_clean = cosine_similarity([saliency_test_img.reshape(-1)], [mean_saliency_normal.reshape(-1)])[0][0]
    cosine_sim_adversarial = cosine_similarity([saliency_test_img.reshape(-1)], [mean_saliency_adv.reshape(-1)])[0][0]
    
    return cosine_sim_clean, cosine_sim_adversarial