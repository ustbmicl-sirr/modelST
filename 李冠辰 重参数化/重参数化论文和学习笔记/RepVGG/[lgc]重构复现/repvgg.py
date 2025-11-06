import torch.nn as nn
import numpy as np
import torch
import copy
import torch.nn.functional as F


class SEBlock(nn.Module):
    """ SEBlock 是著名模型 SENet的核心模块, 能帮助提升性能: 一个模块, 两种激活函数, 学习到特征图的重要性 """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.input_channels = input_channels
        self.down = nn.Conv2d(
            in_channels=input_channels, out_channels=internal_neurons,
            kernel_size=1, stride=1, bias=True
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons, out_channels=input_channels,
            kernel_size=1, stride=1, bias=True
        )

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """ 本函数返回一个卷积层+BN层 """
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        groups=groups, bias=False
    ))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        assert kernel_size == 3
        assert padding == 1
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        padding_11 = padding - kernel_size // 2  # 1×1卷积核该padding几“圈”像素和3×3卷积核大小相等
        self.nonlinearity = nn.ReLU()
        self.se = SEBlock(out_channels, internal_neurons=out_channels // 16) if use_se else nn.Identity()

        if deploy:  # 推理时模型构建: 就一个k×k卷积层
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode
            )
        else:  # 训练时模型构建: 就一个k×k卷积层

            # rbr_identity: stride=1 & 输入输出同维时, 一次残差(含BN)连接
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            # rbr_dense: 3×3卷积 + BN
            self.rbr_dense = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=groups
            )
            # rbr_1x1: 1×1卷积 + BN
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=padding_11, groups=groups
            )

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    def get_custom_L2(self):
        """
        可选的: 1) 用于提升性能  2) 便于量化
            a. 对于 rbr_dense.conv.weight 和 rbr_1x1.conv.weight 不再使用原始权重衰减方案(如 L1, L2等)
            b. 采用 新的权重衰减方案 和 损失函数, 如下:
                loss = criterion(...)
                for every RepVGGBlock blk:
                    loss += weight_decay_coefficient * 0.5 * blk.get_custom_L2()
                optimizer.zero_grad()
                loss.backward()
        """
        K3 = self.rbr_dense.conv.weight  # 3×3卷积核参数
        K1 = self.rbr_1x1.conv.weight  # 1×1卷积核参数
        # t3(3×3卷积核) = γ / (σ + ε)^(0.5)
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        # t1(1×1卷积核) = γ / (σ + ε)^(0.5)
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        # l2_loss_circle —— K3:[C_out, C_in, 3, 3]  和  K3[:, :, 1:2, 1:2]:[C_out, C_in, 1, 1] === 前者的平方和 - 后者的平方和
        # l2_loss_circle —— 相当于不对卷积核中心点进行惩罚约束
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        # eq_kernel —— 卷积核中心点是1×1卷积核和3×3卷积核融合的结果, 其惩罚项单独计算
        # eq_kernel —— 这个中心点是"1×1Conv-BN & 3×3Conv-BN"全部融合后的中心点
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        # eq_kernel —— 标准化, 得到L2_Loss算子
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        # 疑问: 为什么只有中心点计算了"和BN层融合"后的归一化惩罚, 外围却拒绝"和BN层融合", 直接使用权重衰减
        return l2_loss_eq_kernel + l2_loss_circle


    def get_equivalent_kernel_bias(self):
        """ 将3×3卷积-BN, 1×1卷积-BN, (如果有)残差BN进行融合 """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """ 把1×1卷积核的大小扩大为3×3 """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


    def _fuse_bn_tensor(self, branch):
        """ 融合【卷积-BN】 """
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):  # 针对conv-bn层
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:  # 针对纯bn层
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):  # 纯bn层没有卷积核的概念, 为其构造一个单位矩阵充当卷积核
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        # 融合
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    def switch_to_deploy(self):
        """
        使用一个等价代换的 rbr_reparam (卷积层)代替 rbr_dense, rbr_1x1, rbr_identity, id_tensor
        设置 self.deploy = True
        """
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Module):
    """
    RepVGG:
        num_blocks: 一般是长度为4的列表, 代表1+4个stage, 第一个stage固定含一个block, 后4个stage每个共有对应元素个block
            这4个stage, 每个stage的首个卷积操作, 都携带着stride=2参数, 这使得: 1) 降低特征图大小 2) 跳过一次Identity直连
        width_multiplier: 长度也是4, 用于限制/加大宽度, [64, 128, 256, 512] element-wise × [W, X, Y, Z]
        override_groups_map: 用于分组卷积, 详见RepBlock类
        deploy: 定义训练时还是推理时
        use_se: 是否使用SENet中的SE块用于提升性能
    """

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None,
                 override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=3, out_channels=self.in_planes,
            kernel_size=3, stride=2, padding=1,
            deploy=self.deploy, use_se=self.use_se
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(
                in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                stride=stride, padding=1, groups=cur_groups,
                deploy=self.deploy, use_se=self.use_se
            ))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1], num_classes=num_classes,
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_A1(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1], num_classes=num_classes,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_A2(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1], num_classes=num_classes,
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_B0(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_B1(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_B1g2(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map, deploy=deploy
    )


def create_RepVGG_B1g4(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map, deploy=deploy
    )


def create_RepVGG_B2(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_B2g2(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map, deploy=deploy
    )


def create_RepVGG_B2g4(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map, deploy=deploy
    )


def create_RepVGG_B3(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None, deploy=deploy
    )


def create_RepVGG_B3g2(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map, deploy=deploy
    )


def create_RepVGG_B3g4(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map, deploy=deploy
    )


def create_RepVGG_D2se(num_classes=1000, deploy=False):
    return RepVGG(
        num_blocks=[8, 14, 24, 1], num_classes=num_classes,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None, deploy=deploy, use_se=True
    )


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-D2se': create_RepVGG_D2se,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == "__main__":
    model = RepVGGBlock(in_channels=3, out_channels=5, kernel_size=3, padding=1)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            print("=================")
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)
    model.eval()
    inputs = torch.rand((32, 3, 32, 32))
    outputs1 = model(inputs)
    model.switch_to_deploy()
    outputs2 = model(inputs)
    err = ((outputs2 - outputs1)**2).sum()

    print()
