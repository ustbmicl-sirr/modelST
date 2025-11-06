import torch
import torch.nn as nn
import torch.nn.functional as F
from reparameterizer import Reparameterizer


class RepUnit1(Reparameterizer):
    r"""
    A multi-scale and multi-branch feature extraction structure:
    Its inference-time structure is a depthwise conv layer of size 7x7
    Its training-time structure are three parallel depthwise conv layers of size 3x3, 5x5 and 7x7 for depth concatenation
    Args:
        dim (int): Number of input channels.
        deploy: the training-time RepUnit or the inference-time RepUnit. Default: False
    """
    def __init__(self, dim, deploy=False):
        super(RepUnit1, self).__init__()
        assert dim % 3 == 0
        self.dim = dim
        self.deploy = deploy
        self.conv_repara = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
        self.dwconv1 = nn.Conv2d(dim // 3, dim // 3, kernel_size=7, padding=3, groups=dim // 3, bias=False)
        self.dwconv2 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=2, groups=dim // 3, bias=False)
        self.dwconv3 = nn.Conv2d(dim // 3, dim // 3, kernel_size=3, padding=1, groups=dim // 3, bias=False)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)

    def forward(self, x):
        if self.deploy:
            return self.conv_repara(x)
        y1 = self.dwconv1(x[:, :self.dim // 3, :, :])
        y2 = self.dwconv2(x[:, self.dim // 3:2 * self.dim // 3, :, :])
        y3 = self.dwconv3(x[:, 2 * self.dim // 3:, :, :])
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.norm(y)
        return y

    def get_equivalent_parameters(self):
        r"""
        It is called after the training to calculate the parameter of the inference-time structure
        Return: 7x7_dwconv_weight, 7x7_dwconv_bias
        """
        dwconv3_weight = F.pad(self.dwconv3.weight, [2, 2, 2, 2], value=0)
        dwconv2_weight = F.pad(self.dwconv2.weight, [1, 1, 1, 1], value=0)
        dwconv_fused_weight, dwconv_fused_bias = self.depth_concat(
            [self.dwconv1.weight, dwconv2_weight, dwconv3_weight],
            [self.dwconv1.bias, self.dwconv2.bias, self.dwconv3.bias]
        )
        norm_fused_weight, norm_fused_bias = self.fuse_conv_bn_weights(
            dwconv_fused_weight, dwconv_fused_bias,
            self.norm.running_mean, self.norm.running_var,
            self.norm.eps, self.norm.weight, self.norm.bias
        )
        return norm_fused_weight, norm_fused_bias

    def switch_to_deploy(self):
        r"""
        The inference-time structure is established,
        the training-time structure is deleted,
        and the parameters of the inference-time structure are assigned
        Return: None
        """
        all_fused_weight, all_fused_bias = self.get_equivalent_parameters()
        self.deploy = True
        self.__delattr__("dwconv1")
        self.__delattr__("dwconv2")
        self.__delattr__("dwconv3")
        self.__delattr__("norm")
        self.conv_repara.weight.data, self.conv_repara.bias.data = all_fused_weight, all_fused_bias


class RepUnit2(Reparameterizer):
    r"""
    A multi-scale and multi-branch feature extraction structure:
    Its inference-time structure is a depthwise conv layer of size 7x7
    Its training-time structure are three parallel depthwise conv layers of size 3x3, 5x5 and 7x7 for branch addition
    Args:
        dim (int): Number of input channels.
        deploy: the training-time RepUnit or the inference-time RepUnit. Default: False
    """
    def __init__(self, dim, deploy=False):
        super(RepUnit2, self).__init__()
        self.dim = dim
        self.deploy = deploy
        self.conv_repara = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-6)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-6)
        self.norm3 = nn.BatchNorm2d(dim, eps=1e-6)

    def forward(self, x):
        if self.deploy:
            return self.conv_repara(x)
        y1 = self.norm1(self.dwconv1(x))
        y2 = self.norm2(self.dwconv2(x))
        y3 = self.norm3(self.dwconv3(x))
        y = y1 + y2 + y3
        return y

    def get_equivalent_parameters(self):
        r"""
        It is called after the training to calculate the parameter of the inference-time structure
        Return: 7x7_dwconv_weight, 7x7_dwconv_bias
        """
        fused_dwconv1_weight, fused_dwconv1_bias = self.fuse_conv_bn_weights(
            self.dwconv1.weight, self.dwconv1.bias,
            self.norm1.running_mean, self.norm1.running_var,
            self.norm1.eps, self.norm1.weight, self.norm1.bias
        )
        dwconv2_weight, fused_dwconv2_bias = self.fuse_conv_bn_weights(
            self.dwconv2.weight, self.dwconv2.bias,
            self.norm2.running_mean, self.norm2.running_var,
            self.norm2.eps, self.norm2.weight, self.norm2.bias
        )
        dwconv3_weight, fused_dwconv3_bias = self.fuse_conv_bn_weights(
            self.dwconv3.weight, self.dwconv3.bias,
            self.norm3.running_mean, self.norm3.running_var,
            self.norm3.eps, self.norm3.weight, self.norm3.bias
        )
        fused_dwconv2_weight = F.pad(dwconv2_weight, [1, 1, 1, 1], value=0)
        fused_dwconv3_weight = F.pad(dwconv3_weight, [2, 2, 2, 2], value=0)
        dwconv_fused_weight = fused_dwconv1_weight + fused_dwconv2_weight + fused_dwconv3_weight
        dwconv_fused_bias = fused_dwconv1_bias + fused_dwconv2_bias + fused_dwconv3_bias
        return dwconv_fused_weight, dwconv_fused_bias

    def switch_to_deploy(self):
        r"""
        The inference-time structure is established,
        the training-time structure is deleted,
        and the parameters of the inference-time structure are assigned
        Return: None
        """
        all_fused_weight, all_fused_bias = self.get_equivalent_parameters()
        self.deploy = True
        self.__delattr__("dwconv1")
        self.__delattr__("dwconv2")
        self.__delattr__("dwconv3")
        self.__delattr__("norm1")
        self.__delattr__("norm2")
        self.__delattr__("norm3")
        self.conv_repara.weight.data, self.conv_repara.bias.data = all_fused_weight, all_fused_bias
