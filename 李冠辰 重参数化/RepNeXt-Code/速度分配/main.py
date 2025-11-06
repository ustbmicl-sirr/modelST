import torch
import os
import numpy as np
from torchvision.models import resnet18
import time
from convnext import Block
# import RepNeXtV3.Block as repnext_block


# x = torch.rand(64, 192, 32, 32)
# x.cuda()
# model = Block(dim=192, drop_path=0., layer_scale_init_value=1e-6)
# model.cuda
# with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
#     model(x)
# print(prof)
# prof.export_chrome_trace('profiles')


if __name__ == '__main__':
    model = resnet18(pretrained=False)
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,224,224).to(device)

    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace('./resnet_profile')
    
