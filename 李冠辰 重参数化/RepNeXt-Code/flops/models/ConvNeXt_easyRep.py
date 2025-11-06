import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
from timm.models.registry import register_model


class ReparaUnit(nn.Module):
    r"""
    A multiscale feature extraction structure:
    Its inference-time structure is a dw-convolution of size 7x7
    Its training-time structure are three parallel dw-convolutions of size 3x3, 5x5 and 7x7, see Xception...
    """

    def __init__(self, dim, deploy=False):
        super(ReparaUnit, self).__init__()
        # assert dim % 3 == 0
        self.dim = dim
        self.deploy = deploy
        self.conv = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)

    def forward(self, x):
        y = self.norm(self.conv(x))
        return y


class Block(nn.Module):
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
def convnext_easyrep_tiny(pretrained=False, **kwargs):
    return RepNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


@register_model
def convnext_easyrep_small(pretrained=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


@register_model
def convnext_easyrep_base(pretrained=False, in_22k=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@register_model
def convnext_easyrep_large(pretrained=False, in_22k=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@register_model
def convnext_easyrep_xlarge(pretrained=False, in_22k=False, **kwargs):
    return RepNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)


if __name__ == "__main__":
    data = torch.rand((8, 3, 64, 64))
    model = convnext_easyrep_tiny()
    output = model(data)
    print(output.shape)
