# XAIM

# Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)

## Overview
This Python script is designed for computer vision tasks, including loading a pre-trained model, generating attention maps, and classifying images as clean or adversarial.

## Prerequisites

Before using this script, ensure you have the following prerequisites installed:

- Python (>=3.6)
- PyTorch
- matplotlib
- NumPy
- Pillow (PIL)
- torchvision
- captum
- scikit-learn

You can install these dependencies using `pip`:

or using the requirements.txt file:

```bash
pip install -r requirements.txt
```


```bash
pip install torch matplotlib numpy pillow torchvision captum scikit-learn
```

## Usage
<!-- simplify -->
```bash
python main.py --model_path ./vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth \
 --device cuda:3 \
 --attack all \
 --train_path ./data/real_data/train/ \ 
 --num_train_imgs 2000 \
 --num_test_imgs 500 \
 --test_path ./data/real_data/test/ \
 --dataset_class Normal \
 --block -1 \
 --eps 0.05 \
 --force_recompute
```

Command-line Options
- --model_path (default: '/home/aneeshashmi/xai/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth'): Path to the pre-trained model checkpoint.
- --device (default: 'cuda:3'): Specify the device for model execution (e.g., 'cuda:0' or 'cpu').
- --attack (default: 'all'): Specify the type of attack (choices: 'FGSM', 'PGD', 'all').
- --train_path (default: '/home/aneeshashmi/xai/data/real_data/train/'): Path to the training data.
- --num_train_imgs (default: 2000): Number of training images to use.
- --num_test_imgs (default: 500): Number of test images to use.
- --test_path (default: '/home/aneeshashmi/xai/data/real_data/test/'): Path to the test data.
- --dataset_class (default: 'Normal'): Specify the dataset class (choices: 'tb', 'normal').
- --block (default: -1): ViT block to take attention from.
- --eps (default: 0.05): Epsilon for adversarial attacks.
- --force_recompute: Force recompute mean images.


## Results

- The script will generate classification results and save them to the ./results directory. 
- You can find results in CSV format with filenames like preds_dataset_class_block_images_eps.csv
- The script will also generate logs for each run in the ./logs.txt file.
