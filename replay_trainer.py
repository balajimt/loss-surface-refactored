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

# Weight configurations
now = datetime.now()
CURRENT_TIME = now.strftime("%d_%m_%Y_%H_%M_%S")
CURRENT_DAY = now.strftime("%d_%m_%Y")
REPLAY_WEIGHTS = os.path.join("replay", CURRENT_TIME)
NAIVE_WEIGHTS = os.path.join("naive", CURRENT_TIME)
LOG_DIRECTORY = "logs"
Path(REPLAY_WEIGHTS).mkdir(parents=True, exist_ok=True)
Path(NAIVE_WEIGHTS).mkdir(parents=True, exist_ok=True)
Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    filename = os.path.join("logs", current_time + "_logs.log")
    logger.info(str("LOGGING TO" + os.path.abspath(filename)))
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger

logger = setup_logger()

def run_replay(state_dict_location=None, n_vertices=3, train_epochs=20):
    if state_dict_location is None:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    else:
        # SimplexNet(10, VGG16Simplex, n_vert=1, fix_points=fix_pts)
        model = SimplexNet(10, VGG16Simplex, n_vert=n_vertices)
        model.load_state_dict(torch.load(state_dict_location))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = CrossEntropyLoss()
    cl_strategy = Replay(model, optimizer, criterion, train_mb_size=128, train_epochs=train_epochs, eval_mb_size=128, device=device)

    # test_stream = get_tasks(isTrain=False)
    train_stream = get_tasks(isTrain=True)

    count = 0
    for experience in train_stream:
        count += 1
        # if count == 3:
        #     logger.info("Finished REPLAY for task 2")
        #     break
        cl_strategy.train(experience)
        if count == 2:
            # Reinitialize for task 2, but use the same model config as task 1
            # Using replay buffer for task 1
            cl_strategy.model = model

        checkpoint = cl_strategy.model.state_dict()
        fname = os.path.join(REPLAY_WEIGHTS, 'replay' + str(count) + '.pt')
        logger.info("Saving REPLAY for task " + str(count)  + " weights to " + fname)
        torch.save(checkpoint, fname)

    # cl_strategy.train(train_stream)
    checkpoint = cl_strategy.model.state_dict()
    fname = os.path.join(REPLAY_WEIGHTS, 'replay_final.pt')
    logger.info("Saved REPLAY weights to " + fname)
    torch.save(checkpoint, fname)


def run_naive(state_dict_location=None, n_vertices=3, train_epochs=20):
    if state_dict_location is None:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    else:
        model = SimplexNet(10, VGG16Simplex, n_vert=n_vertices)
        model.load_state_dict(torch.load(state_dict_location))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = CrossEntropyLoss()
    cl_strategy = Naive(model, optimizer, criterion, train_mb_size=128, train_epochs=train_epochs, eval_mb_size=128)

    # test_stream = get_tasks(isTrain=False)
    train_stream = get_tasks(isTrain=True)

    count = 0
    for experience in train_stream:
        count += 1
        if count == 3:
            logger.info("Finished NAIVE for task 2")
            break
        cl_strategy.train(experience)
        if count == 2:
            # Reinitialize for task 2, but use the same model config as task 1
            # Using replay buffer for task 1
            cl_strategy.model = model

    # cl_strategy.train(train_stream)
    checkpoint = cl_strategy.model.state_dict()
    fname = os.path.join(NAIVE_WEIGHTS, 'naive.pt')
    torch.save(checkpoint, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")
    parser.add_argument(
        "--state_dict_path",
        default='output_weights/28_10_2022_14_13_12/state_dict_test_28_10_2022_14_13_12.pth',
        help="Path to saved state dict from task 1",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--n_vertices",
        type=int,
        default=6,
        help="Number of vertices of model",
    )
    args = parser.parse_args()
    logger.info("Using config " + str(args))
    logger = setup_logger()
    run_replay(args.state_dict_path, args.n_vertices, args.train_epochs)
    # run_naive(args.state_dict_path, args.n_vertices, args.train_epochs)
