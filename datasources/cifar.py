# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
from torchvision import datasets
from torchvision import transforms

import GLOB as glob
from utils.base.comm import CommUtils as comm


class CIFAR10Data:
    def __init__(self):
        self.dataset_name = "cifar10"
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.imgType = "png"
        self.num_classes = 10
        self.inpRes = 32
        self.outRes = 32

        self.root = "D:/00Data/cifar10(Classification)/data"

    def getSemiData(self, num_labeled, batch_size, mu, iter_num):
        c10 = datasets.CIFAR10(self.root, train=True, download=True)
        labeled_idxs, unlabeled_idxs, labeled_idxs_exp, unlabeled_idxs_exp = self._cache(_split(c10.targets, num_labeled, self.num_classes, batch_size, mu, iter_num), [num_labeled, batch_size, mu, iter_num])  # 训练集样本分割，标记样本、无标记样本

        valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        validDS = datasets.CIFAR10(self.root, train=False, transform=valid_transform, download=True)

        return np.array(labeled_idxs), np.array(unlabeled_idxs), np.array(labeled_idxs_exp), np.array(unlabeled_idxs_exp), self.mean, self.std, validDS

    def _cache(self, dataArray, paramArray):
        saveName = self.dataset_name
        for item in paramArray:
            saveName += "_{}".format(item)
        savePath = "{}/datasources/temp_data/{}.json".format(glob.root, saveName)
        if not comm.file_isfile(savePath):
            comm.json_save(dataArray, savePath, isCover=True)
            return dataArray
        else:
            return comm.json_load(savePath)


class CIFAR100Data:
    def __init__(self):
        self.dataset_name = "cifar100"
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.imgType = "png"
        self.num_classes = 100
        self.inpRes = 32
        self.outRes = 32
        self.root = "D:/00Data/cifar100(Classification)/data"
        pass

    def getSemiData(self, num_labeled, batch_size, mu, iter_num):
        c100 = datasets.CIFAR100(self.root, train=True, download=True)
        labeled_idxs, unlabeled_idxs, labeled_idxs_exp, unlabeled_idxs_exp = self._cache(_split(c100.targets, num_labeled, self.num_classes, batch_size, mu, iter_num), [num_labeled, batch_size, mu, iter_num])  # 训练集样本分割，标记样本、无标记样本

        valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        validDS = datasets.CIFAR100(self.root, train=False, transform=valid_transform, download=True)

        return np.array(labeled_idxs), np.array(unlabeled_idxs), np.array(labeled_idxs_exp), np.array(unlabeled_idxs_exp), self.mean, self.std, validDS

    def _cache(self, dataArray, paramArray):
        saveName = self.dataset_name
        for item in paramArray:
            saveName += "_{}".format(item)
        savePath = "{}/datasources/temp_data/{}.json".format(glob.root, saveName)
        if not comm.file_isfile(savePath):
            comm.json_save(dataArray, savePath, isCover=True)
            return dataArray
        else:
            return comm.json_load(savePath)


def _split(labels, num_labeled, num_classes, batch_size, mu, iter_num):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array([label_idx for label_idx in [item for item in range(len(labels))] if label_idx not in labeled_idx])

    # unlabeled expend
    unlabeled_num = batch_size * mu * iter_num
    unlabeled_num_expand_x = math.ceil(unlabeled_num / len(unlabeled_idx))
    unlabeled_idx_exp = np.hstack([unlabeled_idx for _ in range(unlabeled_num_expand_x)])
    unlabeled_idx_exp = unlabeled_idx_exp[0:unlabeled_num]
    np.random.shuffle(unlabeled_idx_exp)

    # labeled expend
    labeled_num = batch_size * iter_num
    labeled_num_expand_x = math.ceil(labeled_num / len(labeled_idx))
    labeled_idx_exp = np.hstack([labeled_idx for _ in range(labeled_num_expand_x)])
    labeled_idx_exp = labeled_idx_exp[0:labeled_num]
    np.random.shuffle(labeled_idx_exp)

    return labeled_idx.tolist(), unlabeled_idx.tolist(), labeled_idx_exp.tolist(), unlabeled_idx_exp.tolist()
