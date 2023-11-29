import numpy as np
import torch
import torch.nn as nn
from functools import reduce

class NeuralDecisionTree(nn.Module):
    def __init__(self, num_classes, num_cut):
        super(NeuralDecisionTree, self).__init__()
        self.num_leaf = int(np.prod(np.array(num_cut) + 1))
        self.cut_points_list = nn.ParameterList([nn.Parameter(torch.rand([i], requires_grad=True)) for i in num_cut])
        self.leaf_score = nn.Parameter(torch.rand([self.num_leaf, num_classes], requires_grad=True))

    def forward(self, x, temperature=0.1):
        x = x[-1]
        B, T, H = x.size()
        all_outputs = []
        for b in range(B):
            seq = x[b]  
            seq = seq.view(T, H) 

            leaf = reduce(self.torch_kron_prod, map(lambda z: self.torch_bin(seq[:, z[0]:z[0] + 1], z[1], temperature), enumerate(self.cut_points_list)))
            out = torch.matmul(leaf, self.leaf_score)
            all_outputs.append(out)

        # Stack all sequence outputs to form the batch output
        out = torch.stack(all_outputs, dim=0) # [B, T, num_classes]
        out = torch.mean(out, dim=1) # [B, num_classes]
        return out

    @staticmethod
    def torch_kron_prod(a, b):
        res = torch.einsum('ij,ik->ijk', [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    @staticmethod
    def torch_bin(x, cut_points, temperature=0.1):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1]).to("cuda:0")
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(torch.cat([torch.zeros([1]).to("cuda:0"), -cut_points], 0), 0)
        h = torch.matmul(x, W) + b
        res = torch.exp(h - torch.max(h))
        res = res / torch.sum(res, dim=-1, keepdim=True)
        return h