# cl_strategy.eval(test_stream)
import sys

sys.path.append("../../simplex/")
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16Simplex
from simplex_models import SimplexNet

from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.strategies import Naive, CWRStar, Replay
from avalanche.training.strategies import Naive
import torch
from task_splitter import *

import logging
from datetime import datetime
import os
from pathlib import Path
import argparse
from pathlib import Path


def run_naive(state_dict_location='/tmp/loss-surface-refactored/output_weights/29_05_2022_02_35_52/state_dict_test_29_05_2022_02_35_52.pth',
              n_vertices=10, train_epochs=20):
    if state_dict_location is None:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    else:
        model = SimplexNet(10, VGG16Simplex, n_vert=n_vertices)
        model.load_state_dict(torch.load(state_dict_location))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = CrossEntropyLoss()
    cl_strategy = Naive(model, optimizer, criterion, train_mb_size=128, train_epochs=train_epochs, eval_mb_size=128)

    test_stream = get_tasks(isTrain=False)
    cl_strategy.eval(test_stream)


run_naive()
# run_naive(state_dict_location='output_weights/28_05_2022_14_34_20/base_0_simplex_4.pt',n_vertices=5)

