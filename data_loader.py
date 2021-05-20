import math
import torch
import socket
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio

hostname = socket.gethostname()

GOLF_DATA_LISTING = '/srv/bat/data/frames-stable-many/golf.txt'
DATA_ROOT = '/srv/bat/data/frames-stable-many/'


def load_dataset(opt):
    if opt.dataset == 'mmnist':
        from data.moving_mnist import MovingMNIST
        train_data = MovingMNIST(train=True)
    elif opt.dataset == 'kth':
        from data.kth import KTH
        train_data = KTH(train=True)
    elif opt.dataset == 'mazes':
        from data.mazes import DataReader
        data_reader = DataReader(dataset=dataset, time_steps=48)
        train_data = data_reader.provide_dataset()
    return train_data


def sequence_input(seq):
    return [Variable(x.type(dtype)) for x in seq]


def normalize_data(dataset, sequence):
    if dataset == 'mmnist' or dataset == 'kth':
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    return sequence_input(sequence)
