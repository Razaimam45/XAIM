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
from sklearn.metrics import accuracy_score, f1_score

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

def get_reference_attn_matp(model, image_folder, block=-1, n_images=2000, device="cuda", attack_type=["FGSM", "PGD"], select_random=False, plot_path = "./plots/", eps=0.03, random_state=0):
    # print(f"On Block {block}")
    # all_attns, mean_attns, mean_attn_diff
    all_attns, mean_attns, mean_attn_diff = mean_attns_N_images(image_folder=image_folder, n_images=n_images, 
                                                                        block=block, model=model, n_random=select_random, device=device, attack_type=attack_type, eps = eps, random_state=random_state)
    
    for attack_name in attack_type:
        hist_plot(mean_attns['clean'], mean_attns[attack_name], n_images, no_show=True)
        os.makedirs(os.path.join(plot_path, attack_name), exist_ok=True)
        plt.savefig(os.path.join(plot_path, attack_name, f"mean_attns_{attack_name}_block_{args.block}_images_{args.num_train_imgs}_eps_{args.eps}_dataset_{args.dataset_class}.png"))

    return all_attns, mean_attns, mean_attn_diff


def get_model(model_path, device):
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    return model

def get_images_attns(model, image_folder, n_imgs=20, block=-1, device="cuda", attack_type=['all'], eps=0.05, plot=False, random_state=None):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]
    random.seed(random_state)
    image_files = random.sample(image_files, n_imgs)
    all_imgs = []
    all_attns = []
    all_attn_diff = []

    for img_name in image_files:
        img, attns, attn_diff = test_img_attn(
            image_folder=image_folder, 
            block=block, 
            model = model, 
            img_name=img_name, # tested image
            plot=plot, 
            attack_type=attack_type, 
            eps=eps, 
            device=device
            )
        all_imgs.append(img)        
        all_attns.append(attns)
        all_attn_diff.append(attn_diff)

    return all_imgs, all_attns, all_attn_diff, image_files


def classify_image(img_attn_map, mean_attn_clean, mean_attn_adv, method = 'all'): 
    """
        This function classifies an image as clean or adversarial 
        based on the distance between the test image's attention map 
        and the mean attention maps of clean and adversarial images.
    """

    test_attn_flat = img_attn_map.flatten()
    mean_attns_cln_flat = mean_attn_clean.flatten()
    mean_attns_adv_flat = mean_attn_adv.flatten()

    if isinstance(method, str):
        method = [method]

    if "all" in method:
        method = ["sum", "euclidean", "cosine"]

    preds = {}

    if "sum" in method:
        sum_distance_to_normal = np.sum((test_attn_flat - mean_attns_cln_flat))
        sum_distance_to_adversarial = np.sum((test_attn_flat - mean_attns_adv_flat))
        sum_pred = "Clean" if sum_distance_to_normal < sum_distance_to_adversarial else "Adversarial"
        preds["sum"] = sum_pred

    if "euclidean" in method:
        euc_distance_to_normal = euclidean(test_attn_flat, mean_attns_cln_flat)
        euc_distance_to_adversarial = euclidean(test_attn_flat, mean_attns_adv_flat)
        euc_pred = "Clean" if euc_distance_to_normal < euc_distance_to_adversarial else "Adversarial"
        preds["euclidean"] = euc_pred

    if "cosine" in method:
        cosine_distance_to_normal = cosine_similarity([test_attn_flat], [mean_attns_cln_flat])
        cosine_distance_to_adversarial = cosine_similarity([test_attn_flat], [mean_attns_adv_flat])
        cos_pred = "Clean" if cosine_distance_to_normal > cosine_distance_to_adversarial else "Adversarial"
        preds["cosine"] = cos_pred

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/raza.imam/Documents/HC701B/Project/models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth', help='model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--attack', type=str, default='PGD', choices=['FGSM', "PGD", "all"], help='attack type (all for all)')
    parser.add_argument('--train_path', type=str, default="/home/raza.imam/Documents/HC701B/Project/data/TB_data/training", help='path to train data')
    parser.add_argument('--test_path', type=str, default="/home/raza.imam/Documents/HC701B/Project/data/TB_data/testing", help='path to test data')
    parser.add_argument("--num_train_imgs", type=int, default=1000, help="number of train images to use")
    parser.add_argument("--num_test_imgs", type=int, default=100, help="number of test images to use")
    parser.add_argument("--dataset_class", type=str, default="Tuberculosis", choices=["Tuberculosis", "Normal"], help="dataset class")
    parser.add_argument("--block", type=int, default=-1, help="ViT block to take attention from")
    parser.add_argument("--eps", type=float, default=0.03, help="epsilon for adversarial attacks")
    parser.add_argument("--force_recompute", action="store_true", help="force recompute mean images")
    parser.add_argument("--random", default=False, help="select random images for mean attn")
    parser.add_argument("--random_state", type=int, default=0, help="random state for experiments (train and test both)")
    

    args = parser.parse_args()
    print("Arguments:")
    for p in vars(args).items():
        print("  ", p[0]+": ", p[1])


    if args.attack == "all":
        args.attack = ATTACK_LIST # ["PGD", "FGSM"]
    else:
        args.attack = [args.attack.upper()]

    model = get_model(model_path=args.model_path, device=args.device)
    

    if args.force_recompute:
        exits = False
    else:
        # load mean/reference image if it exists
        exits = True
        for attack_name in args.attack + ['clean']:
            # mean_attns[attack_name] = mean_attns[attack_name].cpu().numpy()
            exits *= os.path.exists(os.path.join("./reference", "mean_images", f"mean_attns_{attack_name}_block_{args.block}_images_{args.num_train_imgs}_eps_{args.eps}_dataset_{args.dataset_class}.npy"))

    if exits:
        mean_attns = {}
        for attack_name in args.attack + ['clean']:
            mean_attns[attack_name] = np.load(os.path.join("./reference", "mean_images", f"mean_attns_{attack_name}_block_{args.block}_images_{args.num_train_imgs}_eps_{args.eps}_dataset_{args.dataset_class}.npy"))
        print(f'Loaded mean images from ./reference/mean_images/')
    else:    
        all_attns, mean_attns, mean_attn_diff = get_reference_attn_matp(
            model = model,
            image_folder = os.path.join(args.train_path, args.dataset_class),
            block = args.block,
            n_images = args.num_train_imgs,
            device = args.device,
            attack_type = args.attack,
            select_random = args.random,
            eps = args.eps,
            random_state = args.random_state
        )
    # save mean images to disk
    os.makedirs(os.path.join("./reference", "mean_images"), exist_ok=True)
    print(f'Saving mean images to ./reference/mean_images/')
    for attack_name in args.attack + ['clean']:
        # mean_attns[attack_name] = mean_attns[attack_name].cpu().numpy()
        np.save(os.path.join("./reference", "mean_images", f"mean_attns_{attack_name}_block_{args.block}_images_{args.num_train_imgs}_eps_{args.eps}_dataset_{args.dataset_class}.npy"), mean_attns[attack_name])


    # Get test images
    test_imgs, test_attns, test_attn_diff, test_img_files = get_images_attns(
        model = model,
        image_folder = os.path.join(args.test_path, args.dataset_class),
        n_imgs = args.num_test_imgs,
        block = args.block,
        device = args.device,
        attack_type = args.attack,
        eps = args.eps,
        plot = False,
        random_state = args.random_state,
    )

    num_test_images = len(test_attns)
    # test_attns = attns_pgd

    # classifications = []
    img_name = []
    gt_labels = []
    sum_preds = []
    euc_preds = []
    cos_preds = []

    # Test each test image
    for idx, attn_map in enumerate(test_attns):
        # attn_map = attn_map['PGD']
        # attn_map = attn_map['FGSM']
        for i, attack_name in enumerate(args.attack+['clean']):
            # print(i, attack_name)
            result = classify_image(
                img_attn_map = attn_map[attack_name],
                mean_attn_clean = mean_attns['clean'], 
                mean_attn_adv = mean_attns['PGD'] #WHy PGD hard coded? NOt FGSM?
                )
            sum_preds.append(result['sum'])
            euc_preds.append(result['euclidean'])
            cos_preds.append(result['cosine'])
            gt_labels.append(attack_name)
            img_name.append(test_img_files[idx])

    results_dict ={
        "image": img_name,
        "GT": gt_labels,
        "sum": sum_preds,
        "euclidean": euc_preds,
        "cosine": cos_preds,
    }

    results_df = pd.DataFrame(results_dict)
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv(f"./results/preds_{attack_name}_block_{args.block}_images_{args.num_train_imgs}_eps_{args.eps}_dataset_{args.dataset_class}.csv", index=False)

    # 1 = clean
    # 0 = adversarial
    gt_labels_bin = [1 if label == "clean" else 0 for label in gt_labels]

    methods = ["sum", "euclidean", "cosine"]
    for method in methods:
        pred_bin = [1 if label == "Clean" else 0 for label in results_dict[method]]
        print(f'--------------- {method} ---------------')
        print(f"Accuracy for {method}: {accuracy_score(gt_labels_bin, pred_bin)}")
        print(f"F1 score for {method}: {f1_score(gt_labels_bin, pred_bin)}")
        print(f'-------------------------------------------------------------------')
        print(f'-------------------------------------------------------------------')
    import datetime
    ct = datetime.datetime.now()


    with open("./logs.txt", "a") as f:
        print('', file=f)
        print(f'--------------------------------------------- {ct} ---------------------------------------------\n', file=f)
        print(f'Arguments:\n', file=f)
        for p in vars(args).items():
            print("  ", p[0]+": ", p[1], file=f)
        print(f'--------------------------Results-----------------------------\n', file=f)
        for method in methods:
            pred_bin = [1 if label == "Clean" else 0 for label in results_dict[method]]
            # log to file
            print(f'--------------- {method} ---------------', file=f)
            print(f"Accuracy for {method}: {accuracy_score(gt_labels_bin, pred_bin)}", file=f)
            print(f"F1 score for {method}: {f1_score(gt_labels_bin, pred_bin)}", file=f)
            print(f'-------------------------------------------------------------------', file=f)

            # # print to screen
            # print(f'--------------- {method} ---------------')
            # print(f"Accuracy for {method}: {accuracy_score(gt_labels_bin, pred_bin)}")
            # print(f"F1 score for {method}: {f1_score(gt_labels_bin, pred_bin)}")
            # print(f'-------------------------------------------------------------------')

        print(f'-------------------------------------------------------------------', file=f)
        print(f'', file=f)