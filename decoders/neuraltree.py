# import numpy as np
# from functools import reduce

# import torch
# import torch.nn as nn

# class NeuralDecisionTree(nn.Module):
#     def __init__(
#         self,
#         num_classes,
#         num_cut
#     ):
#         super(NeuralDecisionTree, self).__init__()
#         self.num_leaf = int(np.prod(np.array(num_cut) + 1))
#         self.cut_points_list = nn.ParameterList([nn.Parameter(torch.rand([i], requires_grad=True)) for i in num_cut])
#         self.leaf_score = nn.Parameter(torch.rand([self.num_leaf, num_classes], requires_grad=True))


#     def forward(self, x, temperature=0.1):
#         print(torch.tensor.detach().np.asarray(x).shape)
#         def debug_lambda(z):
#             print(f"Index: {z[0]}, Cut Points: {z[1]}")
#             print(f"Sliced x: {x[:, z[0]:z[0] + 1]}")
#             result = self.torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature)
#             print(f"Result from torch_bin: {result}")
#             return result

#         leaf = reduce(self.torch_kron_prod, map(debug_lambda, enumerate(self.cut_points_list)))

#         # leaf = reduce(self.torch_kron_prod,
#         #               map(lambda z: self.torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(self.cut_points_list)))
#         return torch.matmul(leaf, self.leaf_score)

#     def torch_kron_prod(a, b):
#         res = torch.einsum('ij,ik->ijk', [a, b])
#         res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
#         return res
    
#     def torch_bin(x, cut_points, temperature=0.1):
#         # x is a N-by-1 matrix (column vector)
#         # cut_points is a D-dim vector (D is the number of cut-points)
#         # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
#         D = cut_points.shape[0]
#         W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
#         cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
#         b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0),0)
#         h = torch.matmul(x, W) + b
#         res = torch.exp(h-torch.max(h))
#         res = res/torch.sum(res, dim=-1, keepdim=True)
#         return h
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
        # x = x.view(B * T, H)  # Reshape to 2D tensor for kron prod

        # leaf = reduce(self.torch_kron_prod, map(lambda z: self.torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(self.cut_points_list)))
        # out = torch.matmul(leaf, self.leaf_score)
        # print(out.shape)
        # out = out.view(B, T, -1)  # Reshaped to [B, T, num_classes], might mess up the product?
        # print(out.shape)
        # out = torch.mean(out, dim=1)  # Now shape is [B, num_classes]
        # print(out.shape)
        all_outputs = []
        for b in range(B):
            seq = x[b]  
            seq = seq.view(T, H) 

            leaf = reduce(self.torch_kron_prod, map(lambda z: self.torch_bin(seq[:, z[0]:z[0] + 1], z[1], temperature), enumerate(self.cut_points_list)))
            out = torch.matmul(leaf, self.leaf_score)

            # Optionally aggregate outputs over the sequence
            # For instance, taking the mean or max, or you can design a more complex aggregation mechanism
            # out = torch.mean(out, dim=0)  # Example: mean aggregation over the sequence
            #print(out.shape) T x num_classes
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
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
        h = torch.matmul(x, W) + b
        res = torch.exp(h - torch.max(h))
        res = res / torch.sum(res, dim=-1, keepdim=True)
        return h