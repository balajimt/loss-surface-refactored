import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse
import time
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import sys

sys.path.append("../../simplex/")
import utils

sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from simplex_models import SimplexNet, Simplex
import task_splitter
import logging

import argparse
import tabulate
import utils
import time
from vgg_noBN import SpiralModel
from basic_mlps import BasicSimplex
from simplex_models import SimplexNet
import torch
from datetime import datetime
import os
from pathlib import Path
import logging
from jax import numpy as jnp, random as jr
import numpy as np
import jaxlib
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
    filename = os.path.join("logs", current_time + "_logs.log")
    logger.info("LOGGING TO", os.path.abspath(filename))
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def main(args):
    # Weight configurations
    now = datetime.now()
    CURRENT_TIME = now.strftime("%d_%m_%Y_%H_%M_%S")
    CURRENT_DAY = now.strftime("%d_%m_%Y")
    OUTPUT_WEIGHTS = os.path.join("output_weights", CURRENT_TIME)
    LOG_DIRECTORY = "logs"
    Path(OUTPUT_WEIGHTS).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)
    print("Output weights", OUTPUT_WEIGHTS)
    print("Log directory", LOG_DIRECTORY)

    reg_pars = [0.]
    for ii in range(4, args.n_connector + args.n_mode + 2):
        fix_pts = [True] * (ii)
        start_vert = len(fix_pts)

        out_dim = 10
        simplex_model = SimplexNet(out_dim, VGG16Simplex, n_vert=start_vert,
                                   fix_points=fix_pts)
        simplex_model = simplex_model.cuda()

        log_vol = (simplex_model.total_volume() + 1e-4).log()

        reg_pars.append(max(float(args.LMBD) / log_vol, 1e-8))

    trainloader = task_splitter.get_2task_dataloader(True)
    testloader = task_splitter.get_2task_dataloader(False)
    logger.info(str(["Successfully loaded train and test streams from task number 1 and 2"]))

    fix_pts = [True] * args.n_mode
    n_vert = len(fix_pts)
    complex_ = {ii: [ii] for ii in range(args.n_mode)}
    simplex_model = SimplexNet(10, VGG16Simplex, n_vert=n_vert,
                               simplicial_complex=complex_,
                               fix_points=fix_pts).cuda()

    base_model = VGG16(10)
    base_model = SimplexNet(10, VGG16Simplex, n_vert=9)

    # Import parameters for task 1
    fname =  "output_weights/23_11_2022_14_07_08/state_dict_test_23_11_2022_14_07_08.pth"
    base_model.load_state_dict(torch.load(fname))
    simplex_model.import_base_parameters(base_model, 0)

    # Import parameters for task 2
    fname = "output_weights/23_11_2022_14_07_35/state_dict_test_23_11_2022_14_07_35.pth"
    base_model.load_state_dict(torch.load(fname))
    simplex_model.import_base_parameters(base_model, 1)

    # for ii in range(args.n_mode):
    #     fname = "./saved-outputs/model_" + str(ii) + "/base_model.pt"
    #     base_model.load_state_dict(torch.load(fname))
    #     simplex_model.import_base_parameters(base_model, ii)

    ## add a new points and train ##
    for vv in range(args.n_connector):
        simplex_model.add_vert(to_simplexes=[ii for ii in range(args.n_mode)])
        simplex_model = simplex_model.cuda()
        optimizer = torch.optim.SGD(
            simplex_model.parameters(),
            lr=args.lr_init,
            momentum=0.9,
            weight_decay=args.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        columns = ['vert', 'ep', 'lr', "reg_par", 'tr_loss',
                   'tr_acc', 'te_loss', 'te_acc', 'time', "vol"]

        print(simplex_model.simplicial_complex, flush=True)
        for epoch in range(args.epochs):
            time_ep = time.time()
            if vv == 0:
                train_res = utils.train_epoch_multi_sample(trainloader, simplex_model,
                                                           criterion, optimizer, args.n_sample)
            else:
                train_res = utils.train_epoch_volume(trainloader, simplex_model,
                                                     criterion, optimizer,
                                                     reg_pars[vv], args.n_sample)

            start_ep = (epoch == 0)
            eval_ep = epoch % args.eval_freq == args.eval_freq - 1
            end_ep = epoch == args.epochs - 1
            if start_ep or eval_ep or end_ep:
                test_res = utils.eval(testloader, simplex_model, criterion)
            else:
                test_res = {'loss': None, 'accuracy': None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            values = [vv, epoch + 1, lr, reg_pars[vv],
                      train_res['loss'], train_res['accuracy'],
                      test_res['loss'], test_res['accuracy'], time_ep,
                      simplex_model.total_volume().item()]

            table = tabulate.tabulate([values], columns,
                                      tablefmt='simple', floatfmt='8.4f')
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table, flush=True)

        checkpoint = simplex_model.state_dict()
        fname = os.path.join(OUTPUT_WEIGHTS, str(args.n_mode) +
                             "mode_" + str(vv + 1) + "connector_" + str(args.LMBD) +  CURRENT_TIME + ".pt")
        torch.save(checkpoint, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.005,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "--LMBD",
        type=float,
        default=0.1,
        metavar="lambda",
        help="value for \lambda in regularization penalty",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="verts",
        help="number of vertices in simplex",
    )
    parser.add_argument(
        "--n_mode",
        type=int,
        default=2,
        metavar="N",
        help="number of modes to connect",
    )

    parser.add_argument(
        "--n_connector",
        type=int,
        default=6,
        metavar="N",
        help="number of connecting points to use",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        metavar="N",
        help="how freq to eval test",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        metavar="N",
        help="number of samples to use per iteration",
    )
    args = parser.parse_args()
    logger = setup_logger()
    main(args)
