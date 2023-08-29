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
    
def create_pca_plot(saliency_normal, saliency_adv):
    pca = PCA(n_components=3)

    flat_saliency_normal = saliency_normal
    flat_saliency_adv = saliency_adv

    # pca.fit(flat_saliency_normal)
    pca_saliency_normal = pca.fit_transform(flat_saliency_normal)
    # pca.fit(flat_saliency_adv)
    pca_saliency_adv = pca.fit_transform(flat_saliency_adv)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(pca_saliency_normal[:, 0], pca_saliency_normal[:, 1], pca_saliency_normal[:, 2], c='blue', label='Normal')
    ax1.scatter(pca_saliency_adv[:, 0], pca_saliency_adv[:, 1], pca_saliency_adv[:, 2], c='red', label='Adversarial')
    ax1.set_title('PCA Plot for Saliency (3D): Attn of every image\n1 dot = 1 sample saliency')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')
    ax1.legend()

    plt.tight_layout()
    plt.show()
    
def plot_sal_hist_kde(mean_saliency_normal, mean_saliency_adv, saliency_diff, N_images,\
                        saliencies_normal=None, saliencies_adv=None, kde=True, pca=True):
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
    N = N_images

    axes[0, 0].imshow(mean_saliency_normal, cmap='jet')
    axes[0, 0].set_title(f'Mean ({N} Saliencies of {N}Normals)')

    log_saliency_normal = log_hist(mean_saliency_normal)
    axes[0, 1].hist(log_saliency_normal.flatten(), bins=50, color='blue', alpha=0.7)
    axes[0, 1].set_title('-->Distribution')
    axes[0, 1].set_xlabel('Normalized Log Saliency Value')
    axes[0, 1].set_ylabel('Frequency')

    axes[1, 0].imshow(mean_saliency_adv, cmap='jet')
    axes[1, 0].set_title(f'Mean ({N} Saliencies of {N}Adversarials)')

    log_saliency_adv = log_hist(mean_saliency_adv)
    axes[1, 1].hist(log_saliency_adv.flatten(), bins=50, color='blue', alpha=0.7)
    axes[1, 1].set_title('-->Distribution')
    axes[1, 1].set_xlabel('Normalized Log Saliency Value')
    axes[1, 1].set_ylabel('Frequency')

    axes[2, 0].imshow(saliency_diff, cmap='jet')
    axes[2, 0].set_title('Saliency Difference')

    # Plot the distribution of the difference of saliencies
    print("saliency_diff.shape", saliency_diff.shape)
    log_saliency_diff = log_hist(np.abs(saliency_diff))
    axes[2, 1].hist(log_saliency_diff.flatten(), bins=50, color='green', alpha=0.7)
    axes[2, 1].set_title('-->Distribution')
    axes[2, 1].set_xlabel('Normalized Log Saliency Difference Value')
    axes[2, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
    # kde_plot(mean_saliency_normal.flatten(), mean_saliency_adv.flatten())
    if kde==True:
        kde_plot(log_saliency_normal, log_saliency_adv)
    if pca==True:
        saliencies_normal = np.array(saliencies_normal).reshape(N, -1)
        saliencies_adv = np.array(saliencies_adv).reshape(N, -1)
        create_pca_plot(saliencies_normal, saliencies_adv)
        
def create_pca_plot_MEAN(saliencies_normal, saliencies_adv, test_samples):
    pca = PCA(n_components=3)

    flat_saliency_normal = np.array([matrix.flatten() for matrix in saliencies_normal])
    # print(flat_saliency_normal.shape)
    flat_saliency_adv = np.array([matrix.flatten() for matrix in saliencies_adv])
    # print(flat_saliency_adv.shape)
    flat_saliency_tests = np.array([matrix.flatten() for matrix in test_samples])

    min_max = MinMaxScaler()
    flat_saliency_normal = min_max.fit_transform(flat_saliency_normal)
    flat_saliency_adv = min_max.fit_transform(flat_saliency_adv)
    flat_saliency_tests = min_max.fit_transform(flat_saliency_tests)

    # pca.fit(flat_saliency_normal)
    pca_saliency_normal = pca.fit_transform(flat_saliency_normal)
    pca_saliency_adv = pca.fit_transform(flat_saliency_adv)
    pca_saliency_tests = pca.fit_transform(flat_saliency_tests)
    
    pca_saliency_normal = np.mean(pca_saliency_normal, axis=0)
    pca_saliency_adv = np.mean(pca_saliency_adv, axis=0)
    

    print(pca_saliency_normal)
    # print(pca_saliency_normal)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(pca_saliency_normal[0], pca_saliency_normal[1], pca_saliency_normal[2], c='blue', label='Normal')
    ax1.scatter(pca_saliency_adv[0], pca_saliency_adv[1], pca_saliency_adv[2], c='red', label='Adversarial')
    ax1.scatter(pca_saliency_tests[:,0], pca_saliency_tests[:,1], pca_saliency_tests[:,2], c='green', label='Tests')
    # ax1.set_xticks(np.arange(-10, 10, 2))
    # ax1.set_yticks(np.arange(-10, 10, 2))
    # ax1.set_zticks(np.arange(-10, 10, 2))
    ax1.set_title('PCA Plot for Saliency (3D)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')
    ax1.legend()

    plt.tight_layout()
    plt.show()