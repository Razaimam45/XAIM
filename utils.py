import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from metrics import sum_metric, cosine_similarity_metric, l2_distance_metric, \
    frobenius_norm_metric, kl_divergence_metric, cosine_similarity_1D
import pandas as pd
from collections import Counter


def generate_and_save_metrics(N, image_folder, mean_saliency_normal, mean_saliency_adv, \
                                test_img_saliency, block, head, classifier, metric, out_folder, plot=False, rand=False):
    data = []
    test_saliencies =[]
    for i in range(N):
        image_file, saliency_normal_0, _, _ = test_img_saliency(image_folder, block, head, classifier, plot, rand)
        test_saliencies.append(saliency_normal_0)
        sum_N, sum_Adv = metric(saliency_normal_0, mean_saliency_normal, mean_saliency_adv)
        print(f"sum_del_N: {sum_N}", f"sum_del_A: {sum_Adv}")

        if metric==sum_metric:
            if sum_N <= sum_Adv:
                pred = "Normal"
            else:
                pred = "Adversarial" 
        elif metric==cosine_similarity_metric:
            if sum_N >= sum_Adv:
                pred = "Normal"
            else:
                pred = "Adversarial" 
        elif metric==l2_distance_metric:
            if sum_N <= sum_Adv:
                pred = "Normal"
            else:
                pred = "Adversarial" 
        elif metric==frobenius_norm_metric:
            if sum_N <= sum_Adv:
                pred = "Normal"
            else:
                pred = "Adversarial" 
        elif metric==kl_divergence_metric:
            if sum_N <= sum_Adv:
                pred = "Normal"
            else:
                pred = "Adversarial" 
        elif metric==cosine_similarity_1D:
            threshold = 0.7
            cosine_sim_clean = sum_N
            cosine_sim_adversarial = sum_Adv
            if cosine_sim_clean > threshold:
                pred = "Normal"
            else:
                pred = "Adversarial"
                
        data.append([image_file, sum_N, sum_Adv, pred])
        
    df = pd.DataFrame(data, columns=['Image name', 'sum_N', 'sum_Adv', 'preds'])
    if out_folder!=None:
        df.to_csv(out_folder, index=False)
        
        # df = pd.read_csv(out_folder)
        df['result'] = df.apply(get_results, axis=1)
        print(f'Accuray: {df["result"].mean()}')  
    
    return df

def get_results(row):
    # 1 is for normal
    # 0 is for adversarial
    if row['preds'].lower() == row['Image name'][:6].lower():
        corr = 1
    else:
        corr = 0 
    return corr

def generate_ensemble_preds(N, image_folder, mean_12_saliencies_normal, mean_12_saliencies_adv, test_img_saliency, block, classifier, metric, out_folder=None):
    data = []
    for i in range(N):
        predictions = []
        preds = []
        
        for head in range(len(mean_12_saliencies_normal)):
            
            image_file, saliency_normal_0, _, _ = test_img_saliency(image_folder, block, head, classifier)
            sum_N, sum_Adv = metric(saliency_normal_0, mean_12_saliencies_normal[head], mean_12_saliencies_adv[head])
            
            if metric==sum_metric:
                if sum_N <= sum_Adv:
                    pred = "Normal"
                else:
                    pred = "Adversarial" 
            preds.append(pred)    
        print(f"Predictions for Image {i}: ", preds)
        class_counts = Counter(preds)
        prediction = class_counts.most_common(1)[0][0]
        print(f"Ensemble prediction for image {i}:", prediction)
        
        data.append([image_file, prediction])
    df = pd.DataFrame(data, columns=['Image name', 'preds'])
    df.to_csv(out_folder, index=False)
    
    # df = pd.read_csv(out_folder)
    df['result'] = df.apply(get_results, axis=1)
    print(f'Accuray: {df["result"].mean()}')  
            
    return predictions
