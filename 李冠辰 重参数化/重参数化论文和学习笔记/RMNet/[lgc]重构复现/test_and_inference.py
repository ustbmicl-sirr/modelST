import torch
from resnet2vgg import rmnet20
from data_loader import get_loader
import time
from functools import wraps
from tqdm import tqdm


def time_fn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@time_fn: {fn.__name__} took {t2 - t1: .5f} s")
        return result

    return measure_time


@time_fn
def test(model, dataLoader):
    batch_tail = 0
    test_correct = 0
    for i, (images, labels) in enumerate(tqdm(dataLoader)):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        test_correct += torch.sum(preds == labels.data).to(torch.float32)
        batch_tail += len(labels)
    test_acc = (test_correct / batch_tail).item()
    print(model.__class__.__name__, " 测试准确率为: ", test_acc, end='\t')


# ========= 测试完整模型 =========
test_model_full = rmnet20(num_classes=10)
test_model_full.load_state_dict(torch.load("./mynet.pth"))
test_model_full.eval()
test_model_full.cuda()
test(test_model_full, get_loader()[1])


# ========= 测试等价转换后的模型 =========
test_model_cut = rmnet20(num_classes=10)
test_model_cut.load_state_dict(torch.load("./mynet.pth"))
test_model_cut = test_model_cut.deploy()
test_model_cut.eval()
test_model_cut.cuda()
test(test_model_cut, get_loader()[1])
