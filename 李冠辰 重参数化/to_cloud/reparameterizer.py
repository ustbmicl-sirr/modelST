import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Reparameterizer(nn.Module):
    """
    It encapsulates a set of re-parameterization operations,
    inherited from nn.Module, that can be inherited by network structures that need to be reparameterized
    """
    def __init__(self, *args, **kwargs):
        super(Reparameterizer, self).__init__()

    @staticmethod
    def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
        """
        fuse_conv_bn_weights: a conv layer can be fused with its subsequent bn layer
        Args:
            conv_w: conv weight
            conv_b: conv bias
            bn_rm: bn running mean
            bn_rv: bn running var
            bn_eps: bn eps
            bn_w: bn wright
            bn_b: bn bias
        Return: fused conv weight; fused conv bias
        """
        if conv_b is None:
            conv_b = bn_rm.new_zeros(bn_rm.shape)
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
        return conv_w, conv_b

    @staticmethod
    def trans_bn_2_conv(bn, channels, kernel_size=3, groups=1):
        """
        trans_bn_2_conv: a BN layer can be transformed into a conv layer
        Args:
            bn: the bn layer for transformation
            channels: the number of receiving channels in the bn layer
            kernel_size: the kernel size you want
            groups: the number of groups you want
        Return: fused conv weight; fused conv bias
        """
        assert kernel_size % 2 == 1
        input_dim = channels // groups
        kernel_value = np.zeros((channels, input_dim, kernel_size, kernel_size), dtype=np.float32)
        for i in range(channels):
            kernel_value[i, i % input_dim, kernel_size//2, kernel_size//2] = 1
        kernel = torch.from_numpy(kernel_value).to(bn.weight.device)
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def depth_concat(conv_weights, conv_biases):
        """
        depth_concat: several parallel conv layers for depth concatenation can be merged into a single conv layer
        Args:
            conv_weights: several kernel weights (array)
            conv_biases: several kernel biases (array)
        Return: fused conv weight; fused bias weight
        """
        assert len(conv_weights) == len(conv_biases)
        return torch.cat(conv_weights, dim=0), torch.cat(conv_biases)

    @staticmethod
    def branch_add(conv_weights, conv_biases):
        """
        branch_add: several parallel conv layers for branch addition can be merged into a single conv layer
        Args:
            conv_weights: several kernel weights (array)
            conv_biases: several kernel biases (array)
        Return: fused conv weight; fused bias weight
        """
        assert len(conv_weights) == len(conv_biases)
        fused_weight, fused_bias = conv_weights[0], conv_biases[0]
        for i in range(1, len(conv_weights)):
            fused_weight += conv_weights[i]
            fused_bias += conv_biases[i]
        return fused_weight, fused_bias

    @staticmethod
    def fuse_1x1_kxk_conv(conv_1x1_weight, conv_1x1_bias, conv_kxk_weight, conv_kxk_bias, groups=1):
        """
        fuse_1x1_kxk_conv: merge a pointwise conv layer with its subsequent kxk conv layer into a kxk conv layer
        Note: there is no norm layers between them
        Args:
            conv_1x1_weight: weight of the pointwise conv layer
            conv_1x1_bias: bias of the pointwise conv layer
            conv_kxk_weight: weight of the standard conv layer
            conv_kxk_bias: bias of the standard conv layer
            groups: The number of groups if the kxk conv layer is a groupwise conv layer
        Return: fused kxk conv weight; fused kxk conv bias
        """
        if groups == 1:
            k = F.conv2d(conv_kxk_weight, conv_1x1_weight.permute(1, 0, 2, 3))
            b_hat = (conv_kxk_weight * conv_1x1_bias.reshape(1, -1, 1, 1)).sum((1, 2, 3))
        else:
            k_slices = []
            b_slices = []
            k1_group_width = conv_1x1_weight.size(0) // groups
            k2_group_width = conv_kxk_weight.size(0) // groups
            k1_T = conv_1x1_weight.permute(1, 0, 2, 3)
            for g in range(groups):
                k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
                k2_slice = conv_kxk_weight[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
                k_slices.append(F.conv2d(k2_slice, k1_T_slice))
                b_slices.append((k2_slice * conv_1x1_bias[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
            k, b_hat = Reparameterizer.depth_concat(k_slices, b_slices)
        if conv_kxk_bias is None:
            return k, b_hat
        return k, b_hat + conv_kxk_bias

    @staticmethod
    def fuse_kxk_1x1_conv(conv_kxk_weight, conv_kxk_bias, conv_1x1_weight, conv_1x1_bias, groups=1):
        """
        fuse_kxk_1x1_conv: merge a kxk conv layer with its subsequent pointwise conv layer into a kxk conv layer
        Note: there is no norm layers between them
        Args:
            conv_kxk_weight: weight of the standard conv layer
            conv_kxk_bias: bias of the standard conv layer
            conv_1x1_weight: weight of the pointwise conv layer
            conv_1x1_bias: bias of the pointwise conv layer
            groups: The number of groups if the kxk conv layer is a groupwise conv layer
        Return: fused kxk conv weight; fused kxk conv bias
        """
        if groups == 1:
            k = F.conv2d(conv_kxk_weight.permute(1, 0, 2, 3), conv_1x1_weight).permute(1, 0, 2, 3)
            b_hat = (conv_1x1_weight * conv_kxk_bias.reshape(1, -1, 1, 1)).sum((1, 2, 3))
            return k, b_hat + conv_1x1_bias
        k_slices = []
        b_slices = []
        k1_group_width = conv_1x1_weight.size(0) // groups
        k2_group_width = conv_kxk_weight.size(0) // groups
        k3_T = conv_kxk_weight.permute(1, 0, 2, 3)
        for g in range(groups):
            k1_slice = conv_1x1_weight[:, g * k2_group_width:(g + 1) * k2_group_width, :, :]
            k2_T_slice = k3_T[:, g * k2_group_width:(g + 1) * k2_group_width, :, :]
            k_slices.append(F.conv2d(k2_T_slice, k1_slice))
            b_slices.append((conv_1x1_weight[g * k1_group_width:(g + 1) * k1_group_width, :, :, :] * conv_kxk_bias.reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = Reparameterizer.depth_concat(k_slices, b_slices)
        if conv_1x1_bias is None:
            return k.permute(1, 0, 2, 3), b_hat
        return k.permute(1, 0, 2, 3), b_hat + conv_1x1_bias

    @staticmethod
    def trans_avg_2_conv(channels, kernel_size, groups=1):
        """
        trans_avg_2_conv: an AVG pooling layer can be transformed into a kxk conv layer
        Args:
            channels: number of channels received
            kernel_size: the kernel size of the AVG pooling layer
            groups: The number of groups you want
        Return: the weight of the transformed kxk conv layer
        """
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    @staticmethod
    def lc2pwconv(lc_weight, lc_bias):
        """
        lc2pwconv: a pointwise conv and a fully connected layer can be transformed into each other
        Args:
            lc_weight: the weight of the fully connected layer
            lc_bias: the bias of the fully connected layer
        Return: new conv weight (OxIx1x1), new conv bias (I)
        """
        out_shape, in_shape = lc_weight.shape
        return lc_weight.reshape((out_shape, in_shape, 1, 1)), lc_bias

    @staticmethod
    def lc2pwconv(conv_weight, conv_bias):
        """
        lc2pwconv: a pointwise conv and a fully connected layer can be transformed into each other
        Args:
            conv_weight: the weight of the pointwise conv layer
            conv_bias: the bias of the pointwise conv layer
        Return: new weight of the fully connected layer (OxI), new bias of the fully connected layer (I)
        """
        out_shape, in_shape, _, _ = conv_weight.shape
        return conv_weight.reshape((out_shape, in_shape)), conv_bias
