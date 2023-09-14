import os
import numpy as np
from PIL import Image
import seaborn as sns
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def log_hist(saliency_mean_image):
    saliency_mean_image_flatten = saliency_mean_image.flatten()
    clipped_saliency = (saliency_mean_image_flatten)  # Clip values to -1 and 1
    log_saliency = np.log(np.abs(clipped_saliency) + 1e-5)  # Add a small value to prevent log(0)
    log_saliency = (log_saliency-log_saliency.min())/(log_saliency.max()-log_saliency.min())
    return log_saliency

def kde_plot(log_saliency_normal, log_saliency_adv):
    # Plot the KDE plot
    sns.kdeplot(log_saliency_normal, label='Normal', fill=True)
    sns.kdeplot(log_saliency_adv, label='Adversarial', fill=True)
    plt.ylabel('Density')
    plt.title('Pixel Intensity Distribution (KDE)')
    plt.legend()
    plt.show()
    
# def create_pca_plot(saliency_normal, saliency_adv):
#     pca = PCA(n_components=3)
#     flat_saliency_normal = saliency_normal
#     flat_saliency_adv = saliency_adv
#     # pca.fit(flat_saliency_normal)
#     pca_saliency_normal = pca.fit_transform(flat_saliency_normal)
#     # pca.fit(flat_saliency_adv)
#     pca_saliency_adv = pca.fit_transform(flat_saliency_adv)
#     fig = plt.figure(figsize=(12, 6))
#     ax1 = fig.add_subplot(121, projection="3d")
#     ax1.scatter(pca_saliency_normal[:, 0], pca_saliency_normal[:, 1], pca_saliency_normal[:, 2], c='blue', label='Normal')
#     ax1.scatter(pca_saliency_adv[:, 0], pca_saliency_adv[:, 1], pca_saliency_adv[:, 2], c='red', label='Adversarial')
#     ax1.set_title('PCA Plot for Saliency (3D): Attn of every image\n1 dot = 1 sample saliency')
#     ax1.set_xlabel('Principal Component 1')
#     ax1.set_ylabel('Principal Component 2')
#     ax1.set_zlabel('Principal Component 3')
#     ax1.legend()
#     plt.tight_layout()
#     plt.show()
    
def hist_plot(mean_attns_cln, mean_attns_adv, N_images, no_show = False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    N = N_images
    axes[0, 0].imshow(mean_attns_cln, cmap='inferno')
    axes[0, 0].set_title(f"Mean Attns of {N_images} Cleans")
    log_saliency_normal = log_hist(mean_attns_cln)
    axes[0, 1].hist(log_saliency_normal.flatten(), bins=50, color='green', alpha=0.7)
    axes[0, 1].set_title('-->Distribution')
    axes[0, 1].set_xlabel('Normalized Log Saliency Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[1, 0].imshow(mean_attns_adv, cmap='inferno')
    axes[1, 0].set_title(f"Mean Attns of {N_images} Advs")
    log_saliency_adv = log_hist(mean_attns_adv)
    axes[1, 1].hist(log_saliency_adv.flatten(), bins=50, color='red', alpha=0.7)
    axes[1, 1].set_title('-->Distribution')
    axes[1, 1].set_xlabel('Normalized Log Saliency Value')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    if not no_show:
        plt.show()
    
def plot_statistics(mean_attns_cln, mean_attns_adv, N_images,\
                        attentions_clean=None, attentions_adv=None, hist=True, kde=True, pca=True):
    if hist==True:
        hist_plot(mean_attns_cln, mean_attns_adv, N_images)
    # if kde==True:
    #     kde_plot(log_saliency_normal, log_saliency_adv)
    if pca==True:
        attentions_clean = np.array(attentions_clean).reshape(N_images, -1)
        attentions_adv = np.array(attentions_adv).reshape(N_images, -1)
        create_pca_plot(attentions_clean, attentions_adv)
     
#FIXME:
def create_pca_plot(saliencies_normal, saliencies_adv):
    pca = PCA(n_components=2)

    flat_saliency_normal = np.array([matrix.flatten() for matrix in saliencies_normal])
    # print(flat_saliency_normal.shape)
    flat_saliency_adv = np.array([matrix.flatten() for matrix in saliencies_adv])
    # print(flat_saliency_adv.shape)

    min_max = MinMaxScaler()
    flat_saliency_normal = min_max.fit_transform(flat_saliency_normal)
    flat_saliency_adv = min_max.fit_transform(flat_saliency_adv)

    # pca.fit(flat_saliency_normal)
    pca_saliency_normal = pca.fit_transform(flat_saliency_normal)
    pca_saliency_adv = pca.fit_transform(flat_saliency_adv)
    
    print(pca.explained_variance_ratio_) # Losing 75% variation among the samples
    # pca_saliency_normal = np.mean(pca_saliency_normal, axis=0)
    # pca_saliency_adv = np.mean(pca_saliency_adv, axis=0)
    
    # print(pca_saliency_normal)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    ax1.scatter(pca_saliency_normal[:,0], pca_saliency_normal[:,1], c='blue', label='Normal')
    ax1.scatter(pca_saliency_adv[:,0], pca_saliency_adv[:,1], c='red', label='Adversarial')
    # ax1.set_xticks(np.arange(-10, 10, 2))
    # ax1.set_yticks(np.arange(-10, 10, 2))
    # ax1.set_zticks(np.arange(-10, 10, 2))
    ax1.set_title('PCA Plot for Attns (3D)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    # ax1.set_zlabel('Principal Component 3')
    ax1.legend()

    plt.tight_layout()
    # plt.show()