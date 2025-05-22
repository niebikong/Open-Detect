import argparse
import os
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from model import OpenDetectNet

def get_output(model, data_loader, save_path):
    z = []
    dists = []
    labels = []
    kl_divs = []
    with torch.no_grad():
        for images, label in data_loader:
            images = images.cuda()
            z_iter, dist_iter, kl_div_iter, _ = model(images)
            z.extend(z_iter.cpu().data.numpy())
            dists.extend(dist_iter.cpu().data.numpy())
            kl_divs.extend(kl_div_iter.cpu().data.numpy())
            labels.extend(label.data.numpy())
    z = np.array(z)
    dists = np.array(dists)
    kl_divs = np.array(kl_divs)
    labels = np.array(labels)
    # dist
    dist_reshape = dists.reshape((len(dists), model.n_classes, 1)) 
    dist_class_min = dist_reshape.min(2)  # min dist in each class
    dist_min = np.min(dist_class_min, 1)
    dist_pred = np.argmin(dist_class_min, 1)
    # kl_div
    kld_reshape = kl_divs.reshape((len(dists), model.n_classes, 1)) 
    kld_class_min = kld_reshape.min(2)  # min kld in each class
    kld_min = np.min(kld_class_min, 1)
    kld_pred = np.argmin(kld_class_min, 1)

    np.savez(save_path, z=z, labels=labels, dist_min=dist_min, kld_min=kld_min, dist_pred=dist_pred, kld_pred=kld_pred)
    return z, labels, dist_min, kld_min, dist_pred, kld_pred

def auroc_score(inner_score, open_score, args):  
    y_true = np.array([0] * len(inner_score) + [1] * len(open_score))
    y_score = np.concatenate([inner_score, open_score])
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    opt_threshold = thresholds[maxindex]
    
    y_pred = (y_score >= opt_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print('*'*50)
    print('Open-world AUROC Score')
    print('avg known score: {:.04f}, avg unknown score: {:.04f}, AUROC score {:.04f}'.format(
        np.mean(inner_score), np.mean(open_score), auc_score))
    
    print('Optimal threshold: {:.04f}'.format(opt_threshold))
    print('Accuracy: {:.04f}'.format(accuracy))
    print('Precision: {:.04f}'.format(precision))
    print('Recall: {:.04f}'.format(recall))
    print('F1 Score: {:.04f}'.format(f1))
    print('*'*50)


    return auc_score

def inner_acc(pred_inner, labels_inner):
    inner_corrects = np.sum(pred_inner == labels_inner)
    inner_num = len(labels_inner)
    acc = inner_corrects / inner_num
    print('*'*50)
    print('Closed-world Classification Performance')
    print('inner corrects: {} inner samples: {} inner accuracy {}'.format(
            inner_corrects, inner_num, inner_corrects / inner_num))
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels_inner, pred_inner, average='weighted')
    print('Overall Precision: {:.4f}'.format(precision))
    print('Overall Recall: {:.4f}'.format(recall))
    print('Overall F1 Score: {:.4f}'.format(f1))

    return acc

def test_openset(model, inner_loader, open_loader, args):
    model.eval()
    model = model.cuda()
    inner_save_path = r'/home/ju/Desktop/NetMamba/MGPL/visual/z_data/inner_{}_{}.npz'.format(args.dset, args.split)
    z_inner, inner_label, _, inner_score, _, inner_pred = get_output(model, inner_loader, inner_save_path)
    open_save_path = r'/home/ju/Desktop/NetMamba/MGPL/visual/z_data/open_{}_{}.npz'.format(args.dset, args.split)
    z_open, open_label, _, open_score, _, open_pred = get_output(model, open_loader, open_save_path)
    acc = inner_acc(inner_pred, inner_label)
    auroc = auroc_score(inner_score, open_score, args)
    return acc, auroc


if __name__ == '__main__':
    """
    USTC 10, 3, 2
    mal  10, 1, 6
    combined_USTC_mal 0, 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='USTC', help='dataset') 
    parser.add_argument('--split', type=int, default=10, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    
    DATASETS = [
    'mal',
    'USTC',
    'combined_USTC_mal'
    ]

    setup_seed(2021)
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu

    known_classes, unknown_classes, known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)

    print('Unknown Detection Result')
    print('Dataset: {}    Split: {}'.format(args.dset, args.split))

    inner_set = get_dataset(known_dataset, False, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, False, unknown_classes, 'open')
    print(len(inner_set), len(open_set))
    
    num_samples = min(len(open_set), len(inner_set))
    if len(inner_set) < num_samples:
        raise ValueError(f"inner_set contains only {len(inner_set)} samples, but requested {num_samples}.")
    random_indices = np.random.choice(len(inner_set), size=num_samples, replace=False)
    inner_sampler = SubsetRandomSampler(random_indices)
    inner_loader = DataLoader(inner_set, batch_size=num_samples, sampler=inner_sampler, num_workers=4)


    open_loader = DataLoader(open_set, batch_size=1000, shuffle=False, num_workers=4)
    
    model_dir = './save_model/{}_split_{}.pt'.format(args.dset, args.split)
    model = torch.load(model_dir)

    acc, auroc = test_openset(model, inner_loader, open_loader, args)


