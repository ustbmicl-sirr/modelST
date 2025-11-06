import time
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from timm.models import create_model
from tqdm import trange
from repnext import repnext_u3_tiny, repnext_u3_small, repnext_u3_base
from repnext_middle import repnext_mi_tiny, repnext_mi_small
from repghost import repghostnet_2_0x
from thop import profile, clever_format

def throughput(model, length, optimal_batch_size):
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(device)
    total_time = 0
    for _ in range(20):
        _ = model(dummy_input)
    with torch.no_grad():
        for rep in trange(length):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (length*optimal_batch_size) / total_time
    print("Final Throughput:",Throughput)

def cpu_throughput(model, length, optimal_batch_size):
    device = torch.device("cpu")
    model.to(device)
    dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(device)
    for _ in range(20):
        _ = model(dummy_input)
    starter = time.time()
    with torch.no_grad():
        for _ in trange(length):
            _ = model(dummy_input)
    ender = time.time()
    total_time = (ender - starter)
    Throughput = (length * optimal_batch_size) / total_time
    print("Final Throughput:",Throughput)


def speed(model, length):
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((length,1))
    #GPU-WARM-UP
    for _ in range(20):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in trange(length):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / length
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)


def cpu_speed(model, length):
    device = torch.device("cpu")
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    timings = np.zeros((length,1))
    #GPU-WARM-UP
    for _ in range(20):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in trange(length):
            starter = time.time()
            _ = model(dummy_input)
            ender = time.time()
            curr_time = (ender - starter) * 1000
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / length
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)


def main():
    torch.cuda.set_device(0)
    model = repghostnet_2_0x()  # TODO
    flops, params = profile(model, (torch.randn(1, 3, 224, 224) ,), verbose=False)
    flops, params = clever_format([flops, params])
    print(flops, params)
    # model = create_model("regnety_004")
    # model.structural_reparam()
    # model = create_model("maxvit_tiny_224")
    # for module in model.modules():
    #     if hasattr(module, 'switch_to_deploy'):
    #         module.switch_to_deploy()
    model.convert_to_deploy()
    flops, params = profile(model, (torch.randn(1, 3, 224, 224) ,), verbose=False)
    flops, params = clever_format([flops, params])
    print(flops, params)
    model.eval()
    
    # cpu_speed(model, 100)
    # cpu_throughput(model, 100, 128)

    # speed(model, 1000)
    # throughput(model, 500, 128)
    

if __name__ == "__main__":
    main()
