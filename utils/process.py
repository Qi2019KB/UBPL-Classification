# -*- coding: utf-8 -*-
import torch


class ProcessUtils:
    def __init__(self):
        pass

    @classmethod
    def setVariable(cls, tensor, deviceID, toVariable=True, requires_grad=True):
        if toVariable:
            return torch.autograd.Variable(tensor.to(deviceID, non_blocking=True), requires_grad=requires_grad)
        else:
            return tensor.to(deviceID, non_blocking=True)

    @classmethod
    def features_cov(cls, inp1, inp2, eta=1000):
        bs, n, c, h, w = inp1.size()
        f1 = inp1.clone().view(bs, n, c, h * w)
        f2 = inp2.clone().view(bs, n, c, h * w)
        vecs = torch.stack([f1, f2], -1)
        cov_matrix = cls._torch_cov(cls, vecs)
        return torch.mean(torch.mean(torch.mean(torch.abs(cov_matrix[:, :, :, 0, 1]), dim=-1), dim=-1), dim=-1)*eta, bs*n*c

    def _torch_cov(self, input_vec):
        x = input_vec - torch.mean(input_vec, dim=-2).unsqueeze(-2)
        x_T = torch.transpose(x.clone(), -2, -1)  # [bs, n, c, 2, h*w]
        cov_matrix = torch.matmul(x_T, x) / (x.shape[-2] - 1)
        return cov_matrix
