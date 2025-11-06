import time
import argparse
import os
from thop import profile
from pathlib import Path
# from torchvision.models import mobilenet_v3_small
import torch
from timm.models import create_model
from tqdm import trange
# from models.ConvNeXt_easyRep import *
# from models.RepNeXtV3 import *
import logging


def get_args_parser():
    parser = argparse.ArgumentParser('Params, Flops and Speed Evaluation', add_help=False)
    parser.add_argument('--model', default="convnext_tiny", type=str)
    parser.add_argument('--input_shape', default="1 3 64 64", type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--speed_loop', default=10000, type=int)
    parser.add_argument('--output_path', default="./outputs/log.txt", type=str)
    return parser


def mylogger(file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def flops_and_params(model, input):
    model.eval()
    flops, params = profile(model, inputs=(input, ))
    return flops, params


def speed(model, inputs, length):
    model.cuda()
    inputs = inputs.cuda()
    _ = model(inputs)

    starttime = time.time()
    for i in trange(length):
        y = model(inputs)
    endtime = time.time()

    return endtime - starttime


def get_model(model_name):
    return None


def main(args):
    flops, params, duration = 0, 0 , 0
    logger = mylogger(args.output_path)
    torch.cuda.set_device(args.device_id)
    input_shape = [int(item) for item in args.input_shape.split()]
    inputs = torch.rand(*input_shape)
    model = create_model(args.model)  # TODO
    # model.switch_to_deploy()
    # model = mobilenet_v3_small()
    flops, params = flops_and_params(model, inputs)
    # duration = speed(model, inputs, args.speed_loop)
    logger.info(args.model + " : FLOPS = " + str(flops) + "\t#Params=" + str(params) + "\t" + str(args.speed_loop) + " times inference time = " + str(duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_path and not os.path.isfile(args.output_path):
        file = open(args.output_path,'w')
        file.close()
    main(args)