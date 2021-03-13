# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
import os.path as osp
import pickle
import scipy.stats
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

import data_list
from data_list import ImageList, LoadedImageList, sample_ratios, write_list
import loss
import lr_schedule
import math
import network
import pre_process as prep
import random




def image_classification_test_loaded(test_samples, test_labels, model, device='cpu', num_labels=4):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        len_test = test_labels.shape[0]
        bs = 72
        labels = np.arange(num_labels)
        for i in range(int(len_test / bs)):
            data, target = torch.Tensor(test_samples[bs*i:bs*(i+1), :, :, :]).to(config["device"]), test_labels[bs*i:bs*(i+1)]
            _, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            # Confusion matrix
            try:
                cm += confusion_matrix(target.data.view_as(pred).cpu(), pred.cpu(), labels=labels)
            except:
                cm = confusion_matrix(target.data.view_as(pred).cpu(), pred.cpu(), labels=labels)

        # Last test samples
        data, target = torch.Tensor(test_samples[bs*(i+1):, :, :, :]).to(config["device"]), test_labels[bs*(i+1):]
        _, output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = torch.max(output, 1)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        cm += confusion_matrix(target.data.view_as(pred).cpu(), pred.cpu(), labels=labels)
    print('-----------------------------------------------')
    print(cm)
    per_label_acc = np.diag(cm)/np.sum(cm,1)
    print(per_label_acc, np.sum(per_label_acc)/num_labels)
    print('-----------------------------------------------')
    accuracy = correct / len_test
    test_loss /= len_test * 10
    return accuracy


def test(config):

    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    print("Preparing data", flush=True)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test"]["batch_size"]
    root_folder = data_config["root_folder"]

    dsets["test"] = ImageList(open(osp.join(root_folder, data_config["test"]["list_path"])).readlines(),
                                transform=prep_dict["test"], root_folder=root_folder, ratios=config["ratios_test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                            shuffle=False, num_workers=4)

    test_path = os.path.join(root_folder, data_config["test"]["dataset_path"])
    if os.path.exists(test_path):
        print('Found existing dataset for test', flush=True)
        with open(test_path, 'rb') as f:
            [test_samples, test_labels] = pickle.load(f)
            test_labels = torch.LongTensor(test_labels).to(config["device"])
    else:
        print('Missing test dataset', flush=True)
        print('Building dataset for test and writing to {}'.format(
            test_path), flush=True)
        dset_test = ImageList(open(osp.join(root_folder, data_config["test"]["list_path"])).readlines(),
                                transform=prep_dict["test"], root_folder=root_folder, ratios=config['ratios_test'])
        loaded_dset_test = LoadedImageList(dset_test)
        test_samples, test_labels = loaded_dset_test.samples.numpy(), loaded_dset_test.targets.numpy()
        with open(test_path, 'wb') as f:
            pickle.dump([test_samples, test_labels], f)

    class_num = config["network"]["params"]["class_num"]
    test_samples, test_labels = sample_ratios(
        test_samples, test_labels, config['ratios_test'])

    test_label_distribution = np.zeros((class_num))
    for img in test_labels:
        test_label_distribution[int(img.item())] += 1
    print("Test samples per class: {}".format(test_label_distribution), flush=True)
    test_label_distribution /= np.sum(test_label_distribution)
    print("Test label distribution: {}".format(test_label_distribution), flush=True)

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network.load_state_dict(torch.load(os.path.join(config["save"]),config["method"]+"_{}.pth".format(config["model_weight"])))

    base_network = base_network.to(config["device"])

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    base_network.train(False)
    temp_acc = image_classification_test_loaded(test_samples, test_labels, base_network, num_labels=class_num)
    
    log_str = "  iter: {:05d}, sec: {:.0f}, precision: {:.5f}".format(
        i, time.time() - start_time_test, temp_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, choices=[
                        'NANN', 'DANN', 'IWDAN', 'IWDANORACLE', 'CDAN', 'IWCDAN', 'IWCDANORACLE', 'CDAN-E', 'IWCDAN-E', 'IWCDAN-EORACLE'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"], help="Network type. Only tested with ResNet50")
    parser.add_argument('--dset', type=str, default='VIS_work', choices=['VIS_work', 'office-31', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--test_dset_file', type=str, default='target_test_list.txt', help="The target dataset path list")
    parser.add_argument('--save', type=str, default='save/0000', help="model weight save directory")
    parser.add_argument('--weight', type=str, default='', help="model weight")

    parser.add_argument('--root_folder', type=str, default=None, help="The folder containing the datasets")
    parser.add_argument('--ma', type=float, default=0.5,
                        help='weight for the moving average of iw')
    args = parser.parse_args()

    if args.root_folder is None:
        args.root_folder = 'data/{}/'.format(args.dset)

#   if args.s_dset_file != args.t_dset_file:
    if True:
        # Set GPU ID
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        # Set random number seed.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # train config
        config = {}
        config['method'] = args.method
        config["gpu"] = args.gpu_id
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["save"] = args.save
        if not osp.exists(config["save"]):
            os.system('mkdir -p '+ config["save"])

        config["prep"] = {'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
        if "AlexNet" in args.net:
            config["prep"]['params']['alexnet'] = True
            config["prep"]['params']['crop_size'] = 227
            config["network"] = {"name":network.AlexNetFc, \
                "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, "ma": args.ma} }
        elif "ResNet" in args.net:
            config["network"] = {"name":network.ResNetFc, \
                "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, "ma": args.ma} }
        elif "VGG" in args.net:
            config["network"] = {"name":network.VGGFc, \
                "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, "ma": args.ma} }


        config["dataset"] = args.dset
        config["data"] = {"test": {"list_path": args.test_dset_file, "dataset_path": "{}_test.pkl".format(args.test_dset_file), "batch_size": 4},\
                          "root_folder":args.root_folder}
        config["ratio_test"] = [1]*4 # "4" Should be # of labels
        test(config)
