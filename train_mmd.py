# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os, sys
import os.path as osp
import pickle
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lr_schedule
from data_list import ImageList, LoadedImageList, sample_ratios, image_classification_test_loaded, write_list
from torch.autograd import Variable


optim_dict = {"SGD": optim.SGD}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum // self.count


def transfer_classification(config):

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**prep_config['params'])
    prep_dict["target"] = prep.image_train(**prep_config['params'])
    prep_dict["test"] = prep.image_test(**prep_config['params'])

    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    loss_config = config["loss"]
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}

    ## prepare data
    print("Preparing data", flush=True)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    root_folder = data_config["root_folder"]
    dsets["source"] = ImageList(open(osp.join(root_folder, data_config["source"]["list_path"])).readlines(),
                                transform=prep_dict["source"], root_folder=root_folder, ratios=config["ratios_source"], mode=prep_config['mode'])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(osp.join(root_folder, data_config["target"]["list_path"])).readlines(),
                                transform=prep_dict["target"], root_folder=root_folder, ratios=config["ratios_target"], mode=prep_config['mode'])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(osp.join(root_folder, data_config["test"]["list_path"])).readlines(),
                                transform=prep_dict["test"], root_folder=root_folder, ratios=config["ratios_test"], mode=prep_config['mode'])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs,
                                        shuffle=False, num_workers=4)

    test_path = os.path.join(root_folder, data_config["test"]["dataset_path"])
    if os.path.exists(test_path):
        print('Found existing dataset for test', flush=True)
        with open(test_path, 'rb') as f:
            [test_samples, test_labels] = pickle.load(f)
            test_labels = torch.LongTensor(test_labels).to(device)
    else:
        print('Missing test dataset', flush=True)
        print('Building dataset for test and writing to {}'.format(test_path), flush=True)
        dset_test = ImageList(open(osp.join(root_folder, data_config["test"]["list_path"])).readlines(),
                                transform=prep_dict["test"], root_folder=root_folder, ratios=config['ratios_test'])
        loaded_dset_test = LoadedImageList(dset_test)
        test_samples, test_labels = loaded_dset_test.samples.numpy(
        ), loaded_dset_test.targets.numpy()
        with open(test_path, 'wb') as f:
            pickle.dump([test_samples, test_labels], f)

    class_num = config["network"]["class_num"]
    test_samples, test_labels = sample_ratios(
        test_samples, test_labels, config['ratios_test'])

    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]](**net_config)
    base_network = base_network.to(device)
    if net_config["use_bottleneck"]:
        bottleneck_layer = nn.Linear(base_network.output_num(), net_config["bottleneck_dim"]).to(device)
        classifier_layer = nn.Linear(bottleneck_layer.out_features, class_num)
    else:
        classifier_layer = nn.Linear(base_network.output_num(), class_num)
    for param in base_network.parameters():
        param.requires_grad = False

    classifier_layer = classifier_layer.to(device)

    ## initialization
    if net_config["use_bottleneck"]:
        bottleneck_layer.weight.data.normal_(0, 0.005)
        bottleneck_layer.bias.data.fill_(0.1)
        bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.5))
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)

    ## collect parameters
    if net_config["use_bottleneck"]:
        parameter_list = [{"params":bottleneck_layer.parameters(), "lr":10}, {"params":classifier_layer.parameters(), "lr":10}]

    else:
        parameter_list = [{"params":classifier_layer.parameters(), "lr":10}]

    # compute labels distribution on the source and target domain
    source_label_distribution = np.zeros((class_num))
    for img in dsets["source"].imgs:
        source_label_distribution[img[1]] += 1
    print("Total source samples: {}".format(
        np.sum(source_label_distribution)), flush=True)
    print("Source samples per class: {}".format(
        source_label_distribution), flush=True)
    source_label_distribution /= np.sum(source_label_distribution)
    print("Source label distribution: {}".format(
        source_label_distribution), flush=True)
    target_label_distribution = np.zeros((class_num))
    for img in dsets["target"].imgs:
        target_label_distribution[img[1]] += 1
    print("Total target samples: {}".format(
        np.sum(target_label_distribution)), flush=True)
    print("Target samples per class: {}".format(
        target_label_distribution), flush=True)
    target_label_distribution /= np.sum(target_label_distribution)
    print("Target label distribution: {}".format(
        target_label_distribution), flush=True)
    mixture = (source_label_distribution + target_label_distribution) / 2
    jsd = (scipy.stats.entropy(source_label_distribution, qk=mixture)
           + scipy.stats.entropy(target_label_distribution, qk=mixture)) / 2
    print("JSD : {}".format(jsd), flush=True)
    true_weights = torch.tensor(
        target_label_distribution / source_label_distribution, dtype=torch.float, requires_grad=False)[:, None].to(device)
    write_list(config["out_wei_file"], [round(x, 4) for x in source_label_distribution])
    write_list(config["out_wei_file"], [round(x, 4) for x in target_label_distribution])
    print("True weights : {}".format(true_weights[:, 0].cpu().numpy()))
    config["out_wei_file"].write(str(jsd) + "\n")

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # Maintain two quantities for the QP.
    cov_mat = torch.tensor(np.zeros((class_num, class_num), dtype=np.float32),
                           requires_grad=False).to(device)
    pseudo_target_label = torch.tensor(np.zeros((class_num, 1), dtype=np.float32),
                                       requires_grad=False).to(device)
    # Maintain one weight vector for BER.
    class_weights = torch.tensor(
        1.0 / source_label_distribution, dtype=torch.float, requires_grad=False).to(device)

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    mmd_meter = AverageMeter()
    for i in range(config["num_iterations"]):
        ## test in the train
        if i % config["test_interval"] == 1:
            base_network.train(False)
            classifier_layer.train(False)
            if net_config["use_bottleneck"]:
                bottleneck_layer.train(False)
                test_acc = image_classification_test_loaded(
                    test_samples, test_labels, nn.Sequential(
                        base_network, bottleneck_layer, classifier_layer), device=device)
            else:
                test_acc = image_classification_test_loaded(
                    test_samples, test_labels, nn.Sequential(
                        base_network, classifier_layer), device=device)

            ckpt_path = osp.join(config["save"],config["method"]+"_{}.pth".format(int(temp_acc*100)))
            if torch.cuda.device_count() > 1:
                torch.save(base_network.module.state_dict(), ckpt_path)
            else:
                torch.save(base_network.state_dict(), ckpt_path)

            log_str = 'Iter: %d, mmd = %.4f, test_acc = %.3f' % (
                i, mmd_meter.avg, test_acc)
            print(log_str)
            config["out_log_file"].write(log_str+"\n")
            config["out_log_file"].flush()
            mmd_meter.reset()

        ## train one iter
        if net_config["use_bottleneck"]:
            bottleneck_layer.train(True)
        classifier_layer.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()

        inputs_source, inputs_target, labels_source = Variable(inputs_source).to(device), Variable(inputs_target).to(device), Variable(labels_source).to(device)

        inputs = torch.cat((inputs_source, inputs_target), dim=0)

        features = base_network(inputs)
        if net_config["use_bottleneck"]:
            features = bottleneck_layer(features)

        outputs = classifier_layer(features)

        if 'IW' in loss_config["name"]:
            ys_onehot = torch.zeros(train_bs, class_num).to(device)
            ys_onehot.scatter_(1, labels_source.view(-1, 1), 1)

            # Compute weights on source data.
            if 'ORACLE' in loss_config["name"]:
                weights = torch.mm(ys_onehot, true_weights)
            else:
                weights = torch.mm(ys_onehot, base_network.im_weights)

            source_preds, target_preds = outputs[:train_bs], outputs[train_bs:]
            # Compute the aggregated distribution of pseudo-label on the target domain.
            pseudo_target_label += torch.sum(F.softmax(target_preds, dim=1), dim=0).view(-1, 1).detach()
            # Update the covariance matrix on the source domain as well.
            cov_mat += torch.mm(F.softmax(source_preds,
                                          dim=1).transpose(1, 0), ys_onehot).detach()

            classifier_loss = torch.mean(
                nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                (outputs.narrow(0, 0, inputs.size(0)//2), labels_source) * weights) / class_num
        else:
            classifier_loss = class_criterion(
                outputs.narrow(0, 0, inputs.size(0)//2), labels_source)

        ## switch between different transfer loss
        if loss_config["name"] == "DAN" or loss_config["name"] == "DAN_Linear":
            transfer_loss = transfer_criterion(features.narrow(0, 0, features.size(0)//2), features.narrow(0, features.size(0)//2, features.size(0)//2), **loss_config["params"])
        elif loss_config["name"] == "JAN" or loss_config["name"] == "JAN_Linear":
            softmax_out = nn.Softmax(dim=1)(outputs)
            transfer_loss = transfer_criterion([features.narrow(0, 0, features.size(0)//2), softmax_out.narrow(0, 0, softmax_out.size(0)//2)], [features.narrow(0, features.size(0)//2, features.size(0)//2), softmax_out.narrow(0, softmax_out.size(0)//2, softmax_out.size(0)//2)], **loss_config["params"])
        elif "IWJAN" in loss_config["name"]:
            softmax_out = nn.Softmax(dim=1)(outputs)
            transfer_loss = transfer_criterion([features.narrow(0, 0, features.size(0)//2), softmax_out.narrow(0, 0, softmax_out.size(0)//2)], [features.narrow(0, features.size(0)//2, features.size(0)//2), softmax_out.narrow(0, softmax_out.size(0)//2, softmax_out.size(0)//2)], weights=weights, **loss_config["params"])

        mmd_meter.update(transfer_loss.item(), inputs_source.size(0))
        total_loss = loss_config["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()

        if ('IW' in loss_config["name"]) and i % (config["dataset_mult_iw"] * len_train_source) == config["dataset_mult_iw"] * len_train_source - 1:

            if i > config["dataset_mult_iw"] * len_train_source - 1:
                pseudo_target_label /= train_bs * \
                    len_train_source * config["dataset_mult_iw"]
                cov_mat /= train_bs * len_train_source * config["dataset_mult_iw"]
                print(i, np.sum(cov_mat.cpu().detach().numpy()),
                    train_bs * len_train_source)

                # Recompute the importance weight by solving a QP.
                base_network.im_weights_update(source_label_distribution,
                                               pseudo_target_label.cpu().detach().numpy(),
                                               cov_mat.cpu().detach().numpy(),
                                               device)

            current_weights = [
                round(x, 4) for x in base_network.im_weights.data.cpu().numpy().flatten()]
            write_list(config["out_wei_file"], [np.linalg.norm(
                current_weights - true_weights.cpu().numpy().flatten())] + current_weights)
            print(np.linalg.norm(current_weights -
                                true_weights.cpu().numpy().flatten()), current_weights)

            cov_mat[:] = 0.0
            pseudo_target_label[:] = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('method', type=str, help="loss name",
                        choices=['JAN', 'IWJAN', 'IWJANORACLE', 'JAN_Linear', 'DAN', 'DAN_Linear', 'IWJAN_Linear', 'IWJANORACLE_Linear'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='VIS_work', choices=[
                        'VIS_work','office-31', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--s_dset_file', type=str, nargs='?',
                        default='train_list.txt', help="source data")
    parser.add_argument('--t_dset_file', type=str, nargs='?',
                        default='validation_list.txt', help="target data")
    parser.add_argument('--test_t_dset_file', type=str, default='target_test_list.txt', help="The target dataset path list")
    parser.add_argument('--trade_off', type=float, nargs='?', default=1.0, help="trade_off")
    parser.add_argument('--output_dir', type=str, default='results', help="output directory")
    parser.add_argument('--save', type=str, default='save/0000_0000', help="model weight save directory")
    parser.add_argument('--root_folder', type=str, default=None, help="The folder containing the dataset information")
    parser.add_argument('--seed', type=int, default='42', help="Random seed")
    parser.add_argument('--dataset_mult_iw', type=int, default='0',
                        help="Frequency of weight updates in multiples of the dataset")
    parser.add_argument('--ratio', type=int, default=0, help='ratio option')
    parser.add_argument('--ma', type=float, default=0.5, help='weight for the moving average of iw')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.root_folder is None:
        args.root_folder = 'data/{}/'.format(args.dset)

    if args.s_dset_file == args.t_dset_file:
      sys.exit()

    # Set random number seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = {}
    config["num_iterations"] = 20000
    config["test_interval"] = 500
    config["output_path"] = args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["save"] = args.save
    if not osp.exists(config["save"]):
        os.system('mkdir -p '+ config["save"])
    config["out_log_file"] = open(
        osp.join(config["output_path"], "log.txt"), "w")
    config["out_wei_file"] = open(
        osp.join(config["output_path"], "log_weights.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["prep"] = {'params': {"resize_size":256, "crop_size":224, 'alexnet':False}, 'mode':'RGB'}
    config["loss"] = {"name":args.method, "trade_off":args.trade_off }
    config["data"] = {"source":{"list_path":args.s_dset_file, "batch_size":args.batch_size}, \
                      "target":{"list_path":args.t_dset_file, "batch_size":args.batch_size}, \
                      "test": {"list_path": args.test_t_dset_file, "dataset_path": "{}_test.pkl".format(args.test_t_dset_file), "batch_size": 4}, \
                      "root_folder":args.root_folder}
    config["network"] = {"name":"ResNet50", "use_bottleneck":True, "bottleneck_dim":256, "ma":args.ma}
    config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9, "weight_decay": 0.0005,
                                                           "nesterov": True}, "lr_type": "inv_mmd", "lr_param": {"gamma": 0.0003, "power": 0.75}}

    config["dataset"] = args.dset

    if config["dataset"] == "office-31":
        config["optimizer"]["lr_param"]["init_lr"] = 0.0003
        config["network"]["class_num"] = 31
        config["ratios_source"] = [1] * 31
        if args.ratio == 1:
            config["ratios_source"] = [0.3] * 15 + [1] * 16
        config["ratios_target"] = [1] * 31
        config["ratios_test"] = [1] * 31
        if args.dataset_mult_iw == 0:
            args.dataset_mult_iw = 15
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["network"]["class_num"] = 12
        config["ratios_source"] = [1] * 12
        if args.ratio == 1:
            config["ratios_source"] = [0.3] * 6 + [1] * 6
        config["ratios_target"] = [1] * 12
        config["ratios_test"] = [1] * 12
        if args.dataset_mult_iw == 0:
            args.dataset_mult_iw = 1
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["network"]["class_num"] = 65
        config["ratios_source"] = [1] * 65
        if args.ratio == 1:
            config["ratios_source"] = [0.3] * 32 + [1] * 33
        config["ratios_target"] = [1] * 65
        config["ratios_test"] = [1] * 65
        if args.dataset_mult_iw == 0:
            args.dataset_mult_iw = 15
    else:
        raise ValueError(
            'Dataset cannot be recognized. Please define your own dataset here.')

    config["dataset_mult_iw"] = args.dataset_mult_iw
    config["out_log_file"].write(str(config) + "\n")
    config["out_log_file"].flush()

    print("-" * 50, flush=True)
    print("\nRunning {} on the {} dataset with source {} and target {}\n".format(
        args.method, args.dset, args.s_dset_file, args.t_dset_file), flush=True)
    print("-" * 50, flush=True)

    transfer_classification(config)
