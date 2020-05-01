"""

"""


# Built-in
import os

# Libs
import torch
import numpy as np
from torchsummary import summary

# Own modules


def set_gpu(gpu, enable_benchmark=True):
    """
    Set which gpu to use, also return True as indicator for parallel model if multi-gpu selected
    :param gpu: which gpu(s) to use, could allow a string with device ids separated by ','
    :param enable_benchmark: if True, will let CUDNN find optimal set of algorithms for input configuration
    :return: device instance
    """
    if not isinstance(gpu, str):
        gpu = str(int(gpu))
    if len(str(gpu)) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        parallel = True
        device = torch.device("cuda:{}".format(','.join([str(a) for a in range(len(gpu.split(',')))])))
        print("Devices being used: cuda:", gpu)
    else:
        parallel = False
        device = torch.device("cuda:{}".format(gpu))
        print("Device being used:", device)
    torch.backends.cudnn.benchmark = enable_benchmark
    return device, parallel


def set_random_seed(seed_):
    """
    Set random seed for torch, cudnn and numpy
    :param seed_: random seed to use, could be your lucky number :)
    :return:
    """
    torch.manual_seed(seed_)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_)


def get_model_summary(model, shape, device=None):
    """
    Get model summary with torchsummary
    :param model: the model to visualize summary
    :param shape: shape of the input data
    :return:
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(model.to(device), shape)


if __name__ == '__main__':
    pass
