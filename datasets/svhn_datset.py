# -*- coding: utf-8 -*-
import copy
import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms
from PIL import Image

from utils.randaugment import RandAugmentMC


class SVHN_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mean, std, labeled_indexs, diffAug=True, augCount=1, **kwargs):
        self.svhn = datasets.SVHN(root, split='train', download=False)
        self.labeled_indexs = labeled_indexs
        self.augCount = augCount

        self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.transform_fixMatch = TransformFixMatch(mean=mean, std=std, diffAug=diffAug)

    def __getitem__(self, idx):
        image, target, islabeled = self.svhn.data[idx], self.svhn.labels[idx], 1 if idx in self.labeled_indexs else 0
        image = image.transpose()  # [3, 32, 32] ==> [32, 32, 3]
        image = Image.fromarray(image)  # ndarrary, [32, 32, 3]

        imgs_labeled, imgs_strong, imgs_weak = [], [], []
        for aIdx in range(self.augCount):
            img_labeled = self.transform_labeled(copy.deepcopy(image))
            imgs_labeled.append(img_labeled)
            img_weak, img_strong = self.transform_fixMatch(copy.deepcopy(image))
            imgs_strong.append(img_strong)
            imgs_weak.append(img_weak)

        meta = {"islabeled": islabeled}
        return imgs_labeled, imgs_strong, imgs_weak, target, meta

    def __len__(self):
        return len(self.svhn.labels)


class TransformFixMatch(object):
    def __init__(self, mean, std, diffAug=True):
        self.diffAug = diffAug
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')])

        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        if self.diffAug:
            weak = self.weak(x)
            strong = self.strong(x)
            return self.normalize(weak), self.normalize(strong)
        else:
            weak = self.weak(x)
            weak2 = self.weak2(x)
            return self.normalize(weak), self.normalize(weak2)

