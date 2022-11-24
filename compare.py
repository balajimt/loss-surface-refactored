import os
import argparse
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
import pickle


def setup_logger():
    """
    Function to setup logger. This creates a logger that logs the timestamp along with the provided message.
    The default functionality is to log to both stdout and an offline file in the /logs directory
    :return: logger function
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")

    LOG_DIRECTORY = "logs"
    Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)

    filename = os.path.join("logs", current_time + "_logs.log")
    logger.info("LOGGING TO", os.path.abspath(filename))
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def compare_models(model1, model2):
    """
    Function to compare two models - it first compares model performance on the same task
    and then compares the individual weight values
    :param model1: Model 1
    :param model2: Model 2
    """
    dict1 = collections.defaultdict(float)
    dict2 = collections.defaultdict(float)
    for name, param in model1.named_parameters():
        new_name = name.split("_")[0]
        dict1[new_name] = param
    for name, param in model2.named_parameters():
        new_name = name.split("_")[0]
        dict2[new_name] = param

    return (sum((x - y).abs().sum().pow(2) for x, y in zip(dict1.values(), dict2.values())).pow(0.5))

def get_input_files(input_directory):
    input_files = []
    for i in os.listdir(input_directory):
        if i.endswith('.pt'):
            input_files.append(os.path.join(input_directory, i))
    return input_files

def compare_models_wrapper(folder1, folder2, n_vertices):
    files1 = get_input_files(folder1)
    for file1 in files1:
        name = file1.split('/')[-1]
        file2 = os.path.join(folder2, name)
        model1 = SimplexNet(10, VGG16Simplex, n_vert=n_vertices)
        model1.load_state_dict(torch.load(file1))

        model2 = SimplexNet(10, VGG16Simplex, n_vert=n_vertices)
        model2.load_state_dict(torch.load(file2))
        logger.info(file1 + 'vs' + file2)
        logger.info(str(compare_models(model1, model2)))

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")
    parser.add_argument(
        "--folder1",
        default='/tmp/loss-surface-refactored/output_weights/29_05_2022_02_35_52',
        help="Path to saved state dict from task 1",
    )
    parser.add_argument(
        "--folder2",
        default='/tmp/loss-surface-refactored/output_weights/29_05_2022_02_35_52',
        help="Path to saved state dict from task 1",
    )

    args = parser.parse_args()
    logger = setup_logger()