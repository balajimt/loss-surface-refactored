import torchvision
import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import glob

import tabulate

import os
import sys

sys.path.append("../../simplex/")
import utils
from simplex_helpers import volume_loss
import surfaces
import time

sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from simplex_models import SimplexNet, Simplex
import pickle
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = torchvision.datasets.CIFAR10(args.data_path,
                                       train=True, download=False,
                                       transform=transform_train)