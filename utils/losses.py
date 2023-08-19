# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss.sum(), labels.shape[0]


class ClassDistLoss(nn.Module):
    def __init__(self, scoreThr=0.95):
        super(ClassDistLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.scoreThr = scoreThr

    def forward(self, pred1, pred2):
        assert pred1.size() == pred2.size()
        bs, num_classes = pred1.shape
        pred1_softmax = F.softmax(pred1, dim=1)
        pred2_softmax = F.softmax(pred2, dim=1)
        pred1_score, _ = torch.max(pred1_softmax, dim=-1)
        pred2_score, _ = torch.max(pred2_softmax, dim=-1)
        mask1 = torch.gt(pred2_score, pred1_score).float()
        mask2 = pred2_score.ge(self.scoreThr).float()
        mask = mask1.mul(mask2).unsqueeze(-1)
        loss = self.criterion(pred1_softmax, pred2_softmax) * mask
        return loss.sum() / num_classes, bs, mask.mean(), len([item for item in mask if item > 0])


class ClassFixLoss(nn.Module):
    def __init__(self, scoreThr=0.95):
        super(ClassFixLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.scoreThr = scoreThr

    def forward(self, pred1, pred2):
        pred2_softmax = torch.softmax(pred2, dim=-1)
        max_probs, pseudo_label = torch.max(pred2_softmax, dim=-1)
        mask = max_probs.ge(self.scoreThr).float()
        loss = self.criterion(pred1, pseudo_label) * mask
        n = pred1.shape[0]
        return loss.sum(), n, mask.mean(), len([item for item in mask if item > 0])


class ClassPseudoLoss(nn.Module):
    def __init__(self, is_weight_mean=True, scoreThr=0.95):
        super(ClassPseudoLoss, self).__init__()
        self.scoreThr = scoreThr
        self.is_weight_mean = is_weight_mean
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, preds, targets):
        tgt1_softmax = torch.softmax(targets[0], dim=-1)
        tgt1_score, tgt1_pred = torch.max(tgt1_softmax, dim=-1)
        tgt1_mask = tgt1_score.ge(self.scoreThr).float()

        tgt2_softmax = torch.softmax(targets[1], dim=-1)
        tgt2_score, tgt2_pred = torch.max(tgt2_softmax, dim=-1)
        tgt2_mask = tgt2_score.ge(self.scoreThr).float()

        tgt_fit = torch.eq(tgt1_pred, tgt2_pred).float()
        tgt_mask = tgt_fit.mul(tgt1_mask).mul(tgt2_mask).unsqueeze(-1)
        tgt_weight = F.softmax(torch.stack([tgt1_score, tgt2_score], dim=0), dim=0).unsqueeze(-1)
        tgts_softmax = torch.stack([tgt1_softmax, tgt2_softmax], dim=0)
        tgt_softLabel = torch.mean(tgts_softmax*tgt_weight, dim=0) if self.is_weight_mean else torch.mean(tgts_softmax, dim=0)

        loss = self.criterion(F.softmax(preds, dim=-1), tgt_softLabel) * tgt_mask
        return loss.sum(), preds.shape[0], tgt_mask.mean(), len([item for item in tgt_mask.squeeze() if item > 0])


class ClassSymDistLoss(nn.Module):
    def __init__(self):
        super(ClassSymDistLoss, self).__init__()

    def forward(self, pred1, pred2):
        assert pred1.size() == pred2.size()
        bs, _ = pred1.shape
        num_classes = pred1.size()[1]
        return torch.sum((pred1 - pred2) ** 2) / num_classes, bs


class AvgCounter(object):
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
        self.avg = 0. if self.count == 0 else self.sum / self.count


class AvgCounters(object):
    def __init__(self, num=1):
        self.counters = [AvgCounter() for i in range(num)]
        self.reset()

    def reset(self):
        for counter in self.counters:
            counter.reset()

    def update(self, idx, val, n=1):
        self.check_idx(idx)
        self.counters[idx].update(val, n)

    def avg(self):
        return [item.avg for item in self.counters]

    def sum(self):
        return [item.sum for item in self.counters]

    def check_idx(self, idx):
        if len(self.counters) < idx + 1:
            for i in range(len(self.counters), idx + 1):
                self.counters.append(AvgCounter())


