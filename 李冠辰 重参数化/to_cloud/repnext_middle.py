import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from reparameterizer import Reparameterizer


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.act(input) + self.drop_path(x)
        return x


class RepNeXt(Reparameterizer):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.deploy_blocks = []
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=False)
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
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


def repnext_mi_tiny(pretrained=False, **kwargs):
    model = RepNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = torch.load("Path of the pre-trained model", map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


def repnext_mi_small(pretrained=False, **kwargs):
    model = RepNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = torch.load("Path of the pre-trained model", map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model

def repnext_mi_base(pretrained=False, **kwargs):
    model = RepNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        checkpoint = torch.load("Path of the pre-trained model", map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model