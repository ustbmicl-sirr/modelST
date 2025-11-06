import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from reparameterizer import Reparameterizer


class RepUnit(Reparameterizer):
    r"""
    A multi-scale and multi-branch feature extraction structure:
    Its inference-time structure is a depthwise conv layer of size 7x7
    Its training-time structure are three parallel depthwise conv layers of size 3x1, 1x5 and 7x7 for branch addition
    Args:
        dim (int): Number of input channels.
        deploy: the training-time RepUnit or the inference-time RepUnit. Default: False
    """
    def __init__(self, dim, deploy=False):
        super(RepUnit, self).__init__()
        self.dim = dim
        self.deploy = deploy
        self.conv_repara = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, bias=False)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)
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
        fused_dwconv2_weight = F.pad(dwconv2_weight, [1, 1, 3, 3], value=0)
        fused_dwconv3_weight = F.pad(dwconv3_weight, [3, 3, 2, 2], value=0)
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


class Block(Reparameterizer):
    r"""
    Block:
    DwConv (RepUnit) -> BatchNorm -> 1x1 Conv -> RELU -> 1x1 Conv & activated shortcut branch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        deploy: the training-time block or the inference-time block. Default: False
    """
    def __init__(self, dim, drop_path=0., deploy=False):
        super().__init__()
        self.deploy = deploy
        self.block = None
        self.dim = dim
        self.feature = RepUnit(dim=dim, deploy=deploy)
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
        Shortcut re-parameterization
        Return: None
        """
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


down_sample_flow = 1  # control the re-parameterization of the downsampling layer


class RepNeXt(Reparameterizer):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0.):
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

        self.stages = nn.ModuleList()
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
        Use recursive traversal for the entire network structure
        to re-parameterize the components that satisfy the re-parameterization rules
        Return: None
        """
        def foo(net):
            children = list(net.children())
            if isinstance(net, Block):
                net.feature.switch_to_deploy()
                net.switch_to_deploy()
            elif isinstance(net, nn.Sequential) and isinstance(net[0], nn.Conv2d) and len(net) == 2:
                id_structure = nn.Conv2d(
                    net[0].in_channels, net[0].out_channels,
                    kernel_size=net[0].kernel_size, stride=net[0].stride, padding=net[0].padding,
                    groups=net[0].groups, bias=True).eval()
                fused_weight, fused_bias = self.fuse_conv_bn_weights(
                    net[0].weight, net[0].bias,
                    net[1].running_mean, net[1].running_var,
                    net[1].eps, net[1].weight, net[1].bias
                )
                id_structure.weight.data, id_structure.bias.data = fused_weight, fused_bias
                self.downsample_layers[0] = id_structure
            elif isinstance(net, nn.Sequential) and isinstance(net[1], nn.Conv2d) and len(net) == 2:
                id_structure = nn.Conv2d(
                    net[1].in_channels, net[1].out_channels,
                    kernel_size=net[1].kernel_size, stride=net[1].stride, padding=net[1].padding,
                    groups=net[1].groups, bias=True).eval()
                pw_weight, pw_bias = self.trans_bn_2_conv(net[0], net[0].num_features, kernel_size=1, groups=1)
                fused_weight, fused_bias = self.fuse_1x1_kxk_conv(pw_weight, pw_bias, net[1].weight, net[1].bias, groups=1)
                id_structure.weight.data, id_structure.bias.data = fused_weight, fused_bias
                global down_sample_flow
                self.downsample_layers[down_sample_flow] = id_structure
                down_sample_flow += 1
            else:
                for c in children:
                    foo(c)

        foo(self.eval())


@register_model
def repnext_u3_tiny(pretrained=False, **kwargs):
    model = RepNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = torch.load("Path of the pre-trained model", map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def repnext_u3_small(pretrained=False, **kwargs):
    model = RepNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = torch.load("Path of the pre-trained model", map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def repnext_u3_base(pretrained=False, **kwargs):
    model = RepNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        checkpoint = torch.load("Path of the pre-trained model", map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model
