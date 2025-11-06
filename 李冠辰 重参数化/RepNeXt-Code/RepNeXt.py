import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
from timm.models.registry import register_model


class Reparameterizer(nn.Module):
    r"""
    It encapsulates a set of reparameterization operations,
    inherited from nn.Module, that can be inherited by network structures that need to be reparameterized
    """

    def __init__(self, *args, **kwargs):
        super(Reparameterizer, self).__init__()

    @staticmethod
    def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
        r"""
        The convolution layer is fused with the bn layer that follows
        :param conv_w: conv weight
        :param conv_b: conv bias
        :param bn_rm: bn running mean
        :param bn_rv: bn running var
        :param bn_eps: bn eps
        :param bn_w: bn wright
        :param bn_b: bn bias
        :return: new conv weight; new conv bias
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
        rThe BN layer is transformed into a convolution layer with uniform action
        :param kernel_size: Transform BN into Conv with shape of kernel_size
        :param bn: LAYER
        :param channels: The number of receiving channels at bn layer
        :param groups: The number of groups of grouping convolution
        :return: new conv weight; new conv bias
        """
        input_dim = channels // groups
        kernel_value = np.zeros((channels, input_dim, kernel_size, kernel_size), dtype=np.float32)
        for i in range(channels):
            kernel_value[i, i % input_dim, 1, 1] = 1
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
        r"""
        Several convolution kernels of the same size are spliced in the channel dimension
        :param conv_weights: many kernel weights -> array
        :param conv_biases: many kernel biases -> array
        :return: new conv weight; new bias weight
        """
        assert len(conv_weights) == len(conv_biases)
        fused_biases = torch.cat(conv_biases) if conv_biases[0] else None
        return torch.cat(conv_weights, dim=0), fused_biases

    @staticmethod
    def fuse_1x1_kxk_conv(conv_1x1_weight, conv_1x1_bias, conv_kxk_weight, conv_kxk_bias, groups=1):
        r"""
        Merge a 1x1 convolution with a subsequent kxk convolution into a kxk convolution
        Note: there is no BN layer in between them
        :param conv_1x1_weight: weight of conv_1x1
        :param conv_1x1_bias: bias of conv_1x1
        :param conv_kxk_weight: weight of conv_kxk
        :param conv_kxk_bias: bias of conv_1x1
        :param groups: The number of groups of grouping convolution
        :return: new conv_kxk weight; new conv_kxk bias
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
                b_slices.append(
                    (k2_slice * conv_1x1_bias[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum(
                        (1, 2, 3)))
            k, b_hat = Reparameterizer.depth_concat(k_slices, b_slices)
        if conv_kxk_bias is None:
            return k, b_hat
        return k, b_hat + conv_kxk_bias

    @staticmethod
    def fuse_kxk_1x1_conv(conv_kxk_weight, conv_kxk_bias, conv_1x1_weight, conv_1x1_bias, groups=1):
        r"""
        Merge a kxk convolution with a subsequent 1x1 convolution into a kxk convolution
        Note: there is no BN layer in between them
        :param conv_1x1_weight: weight of conv_1x1
        :param conv_1x1_bias: bias of conv_1x1
        :param conv_kxk_weight: weight of conv_kxk
        :param conv_kxk_bias: bias of conv_1x1
        :param groups: The number of groups of grouping convolution
        :return: new conv_kxk weight; new conv_kxk bias
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
            b_slices.append((conv_1x1_weight[g * k1_group_width:(g + 1) * k1_group_width, :, :,
                             :] * conv_kxk_bias.reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = Reparameterizer.depth_concat(k_slices, b_slices)
        if conv_1x1_bias is None:
            return k.permute(1, 0, 2, 3), b_hat
        return k.permute(1, 0, 2, 3), b_hat + conv_1x1_bias

    @staticmethod
    def trans_avg_2_conv(channels, kernel_size, groups=1):
        r"""
        An AVG pooling layer is transformed into a 3x3 convolution kernel
        :param channels: number of channels received
        :param kernel_size: the kernel size of the AVG pooling layer
        :param groups: The number of groups of grouping convolution
        :return: the weight of the new 3x3 conv
        """
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k


class ReparaUnit(Reparameterizer):
    r"""
    A multiscale feature extraction structure:
    Its inference-time structure is a dw-convolution of size 7x7
    Its training-time structure are three parallel dw-convolutions of size 3x3, 5x5 and 7x7, see Xception...
    """

    def __init__(self, dim, deploy=False):
        super(ReparaUnit, self).__init__()
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
        :return: 7x7_dwconv_weight, 7x7_dwconv_bias
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
        :return: None
        """
        all_fused_weight, all_fused_bias = self.get_equivalent_parameters()
        self.deploy = True
        self.__delattr__("dwconv1")
        self.__delattr__("dwconv2")
        self.__delattr__("dwconv3")
        self.__delattr__("norm")
        self.conv_repara.weight.data, self.conv_repara.bias.data = all_fused_weight, all_fused_bias


class Block(Reparameterizer):
    r"""
    ReparaConvNeXt Block:
    DwConv -> BatchNorm (channels_first) -> 1x1 Conv -> RELU -> 1x1 Conv; all in (N, C, H, W)
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0., deploy=False):
        super().__init__()
        self.deploy = deploy
        self.block = None
        self.dim = dim
        self.feature = ReparaUnit(dim=dim, deploy=deploy)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.deploy:
            return self.block(x)
        y = self.feature(x)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)
        return self.act(x) + self.drop_path(y)

    def switch_to_deploy(self):
        r"""
        Block branch merge
        :return: new serial substructure
        """
        assert self.feature.conv_repara.groups in (1, self.dim)
        id_feature = nn.Conv2d(self.dim, self.dim * 2, kernel_size=7, padding=3, groups=self.dim, bias=True).eval()
        for i in range(id_feature.in_channels):
            nn.init.dirac_(id_feature.weight.data[2 * i: 2 * i + 1])
        id_feature.bias.data[::2] = 0
        id_feature.weight.data[1::2] = self.feature.conv_repara.weight.data
        id_feature.bias.data[1::2] = self.feature.conv_repara.bias.data

        id_pwconv1 = nn.Conv2d(self.dim * 2, self.dim * 5, kernel_size=1, bias=True).eval()
        nn.init.dirac_(id_pwconv1.weight.data[:self.dim * 2:2, ::2])
        nn.init.zeros_(id_pwconv1.weight.data[1:self.dim * 2:2, ::2])
        nn.init.zeros_(id_pwconv1.weight.data[:self.dim * 2:2, 1::2])
        id_pwconv1.bias.data[:self.dim * 2:2] = 0
        id_pwconv1.bias.data[1:self.dim * 2:2] = self.pwconv1.bias.data[:self.dim]
        id_pwconv1.weight.data[1:self.dim * 2:2, 1::2] = self.pwconv1.weight.data[:self.dim]
        nn.init.zeros_(id_pwconv1.weight.data[self.dim * 2:, ::2])
        id_pwconv1.weight.data[self.dim * 2:, 1::2] = self.pwconv1.weight[self.dim:]
        id_pwconv1.bias.data[self.dim * 2:] = self.pwconv1.bias[self.dim:]

        id_pwconv2 = nn.Conv2d(self.dim * 5, self.dim, kernel_size=1, bias=True).eval()
        nn.init.dirac_(id_pwconv2.weight.data[:, :self.dim * 2:2])
        id_pwconv2.weight.data[:, 1:self.dim * 2:2] = self.pwconv2.weight[:, :self.dim]
        id_pwconv2.weight.data[:, self.dim * 2:] = self.pwconv2.weight[:, self.dim:]
        id_pwconv2.bias.data = self.pwconv2.bias

        self.deploy = True
        self.block = nn.Sequential(id_feature, id_pwconv1, self.act, id_pwconv2)
        self.__delattr__("feature")
        self.__delattr__("pwconv1")
        self.__delattr__("pwconv2")
        self.__delattr__("act")
        self.__delattr__("drop_path")


class RepNeXt(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """

    def __init__(
        self, in_chans=3, num_classes=1000, pretrained=False,
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.
    ):
        super().__init__()
        self.deploy_blocks = []
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0])
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(dims[-1], num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x

    def switch_to_deploy(self):
        """
        Use recursive traversal of the entire network structure
        to reparameterize the components that satisfy the reparameterization rules
        :return: The new structure after reparameterization (not a direct replacement) -> nn.Sequential(*self.deploy_blocks)
        """

        def foo(net):
            children = list(net.children())
            if isinstance(net, Block):
                net.feature.switch_to_deploy()
                net.switch_to_deploy()
            else:
                for c in children:
                    foo(c)

        foo(self.eval())


@register_model
def repnext_tiny(pretrained=False, **kwargs):
    return RepNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


@register_model
def repnext_small(pretrained=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


@register_model
def repnext_base(pretrained=False, in_22k=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@register_model
def repnext_large(pretrained=False, in_22k=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@register_model
def repnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)


if __name__ == "__main__":
    data = torch.rand((8, 3, 64, 64))
    model = repnext_tiny()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.2)
            nn.init.uniform_(module.weight, 0, 0.3)
            nn.init.uniform_(module.bias, 0, 0.4)
    model.eval()
    output1 = model(data)
    model.switch_to_deploy()
    output2 = model(data)
    print(((output1 - output2) ** 2).sum())
    flag = torch.allclose(output1, output2)
    print()
