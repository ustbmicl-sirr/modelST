import torch
import torch.nn as nn 
from repnext import repnext_u3_tiny
import numpy as np
from tqdm import trange
import argparse


def speed(model, length=1000, input_shape=(1, 3, 224, 224)):
    """
    speed: record the latency for a model
    Args:
        model: the model for speed test, no need to switch to its eval mode
        length: number of consecutive speed measurements, longer length means more informative test results. Default: 1000
        input_shape: The shape of the input data. Defaults: (1, 3, 224, 224)
    Return: None
    """
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(*input_shape, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((length, 1))
    for _ in range(10):
        _ = model(dummy_input)
    with torch.no_grad():
        for idx in trange(length):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[idx] = curr_time
    mean_syn = np.sum(timings) / length
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print('Mean {mean_syn:.3f}ms Std {std_syn:.3f}ms FPS {mean_fps:.3f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))


def chaos_initialization(model):
    """
    chaos_initialization: random initialization of key parameters of a network
    Args:
        model: the model for initialization
    Return: random initialized model
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.2)
            nn.init.uniform_(module.weight, 0, 0.3)
            nn.init.uniform_(module.bias, 0, 0.4)
    return model


def pretrain_initialization(model, pretrain_root):
    """
    pretrain_initialization: load pre-training parameters for a network
    Args:
        model: the model for initialization
        pretrain_root: storage location for pre-training parameters
    Return: pre-trained model
    """
    checkpoint = torch.load(pretrain_root, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    return model


def consistency_comparation():
    dummy_input = torch.rand(1, 3, 224, 224)
    net = repnext_u3_tiny()
    net = chaos_initialization(net)
    # You may wish to use pre-trained for initialization:
    #     net = pretrain_initialization(net, "Path of the pre-trained model")
    net.eval()
    output_before_rep = net(dummy_input)
    net.switch_to_deploy()
    net.eval()
    output_after_rep = net(dummy_input)
    print("The difference in the network output before and after structural re-parameterization is ",
          ((output_before_rep - output_after_rep) ** 2).sum().data)
    

def speed_comparation():
    net = repnext_u3_tiny()
    net.eval()
    print("Let's see how fast the inference of the training-time RepNeXt-u3-tiny is ... ")
    speed(net)
    net.switch_to_deploy()
    print("Let's see how fast the inference of the inference-time RepNeXt-u3-tiny is ... ")
    speed(net)


if __name__ == "__main__":
    helper = "speed_comparation or consistency_comparation, the two can not be ran together because of conflicts and errors"
    parser = argparse.ArgumentParser("Structural Re-parameterization Validation")
    parser.add_argument('--operation', type=str, default="speed_comparation", help=helper)
    args = parser.parse_args()
    if args.operation == "speed_comparation":
        speed_comparation()
    elif args.operation == "consistency_comparation":
        consistency_comparation()
