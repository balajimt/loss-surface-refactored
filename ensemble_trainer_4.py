import sys

sys.path.append("../../simplex/")
sys.path.append("../../simplex/models/")

import argparse
import tabulate
import utils
import time
from vgg_noBN import VGG16Simplex
from simplex_models import SimplexNet
import torch
import task_splitter
from datetime import datetime
import os
from pathlib import Path
import logging

# Weight configurations
now = datetime.now()
CURRENT_TIME = now.strftime("%d_%m_%Y_%H_%M_%S")
CURRENT_DAY = now.strftime("%d_%m_%Y")
OUTPUT_WEIGHTS = os.path.join("output_weights", CURRENT_TIME)
LOG_DIRECTORY = "logs"
Path(OUTPUT_WEIGHTS).mkdir(parents=True, exist_ok=True)
Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)

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
    global simplex_model
    savedir = "./saved-outputs/"

    ## randomly initialize simplexes to determine regularization parameters ##
    reg_pars = []
    for ii in range(args.n_verts + 1):
        fix_pts = [True] * (ii + 1)
        start_vert = len(fix_pts)

        out_dim = 10
        simplex_model = SimplexNet(out_dim, VGG16Simplex, n_vert=start_vert,
                                   fix_points=fix_pts)
        # simplex_model = simplex_model.cuda()

        log_vol = (simplex_model.total_volume() + 1e-4).log()

        reg_pars.append(max(float(args.LMBD) / log_vol, 1e-8))

    ## import training and testing data ##
    task_number = args.task_number
    trainloader = task_splitter.get_task_dataloader(task_number, True)
    testloader = task_splitter.get_task_dataloader(task_number, False)
    logger.info(str(["Successfully loaded train and test streams from task number", task_number]))

    # Simplex code starts here
    columns = ['component', 'vert', 'ep', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', "vol"]
    criterion = torch.nn.CrossEntropyLoss()
    for component in range(args.n_component):
        fix_pts = [False]
        if args.model is None:
            simplex_model = SimplexNet(10, VGG16Simplex, n_vert=1, fix_points=fix_pts)
        else:
            print(args.n_verts)
            simplex_model = SimplexNet(10, VGG16Simplex, n_vert=args.n_verts)
            simplex_model.load_state_dict(torch.load(args.model))
        # simplex_model = SimplexNet(10, VGG16Simplex, n_vert=1, fix_points=fix_pts)
        for vv in range(args.n_verts):
            if vv == 0:
                optimizer = torch.optim.SGD(
                    simplex_model.parameters(),
                    lr=args.base_lr,
                    momentum=0.9,
                    weight_decay=args.wd
                )
            else:
                optimizer = torch.optim.SGD(
                    simplex_model.parameters(),
                    lr=args.simplex_lr,
                    momentum=0.9,
                    weight_decay=args.wd
                )

            n_epoch = args.base_epochs if vv == 0 else args.simplex_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=n_epoch)

            for epoch in range(n_epoch):
                time_ep = time.time()
                if vv == 0:
                    train_res = utils.train_epoch_avalanche(trainloader, simplex_model,
                                                            criterion, optimizer)
                else:
                    train_res = utils.train_epoch_volume_avalanche(trainloader, simplex_model,
                                                                   criterion, optimizer,
                                                                   reg_pars[vv], args.n_sample)

                start_ep = (epoch == 0)
                eval_ep = epoch % args.eval_freq == args.eval_freq - 1
                end_ep = epoch == n_epoch - 1
                if start_ep or eval_ep or end_ep:
                    test_res = utils.eval_avalanche(testloader, simplex_model, criterion)
                else:
                    test_res = {'loss': None, 'accuracy': None}

                time_ep = time.time() - time_ep

                lr = optimizer.param_groups[0]['lr']
                scheduler.step()

                values = [component, vv, epoch + 1,
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
                logger.info(table)
                checkpoint = simplex_model.state_dict()
                fname = "base_" + str(component) + "_epoch_" + str(epoch) + "_simplex_" + str(vv) + ".pt"
                torch.save(checkpoint, os.path.join(OUTPUT_WEIGHTS, fname))

            checkpoint = simplex_model.state_dict()
            fname = "base_" + str(component) + "_simplex_" + str(vv) + ".pt"
            torch.save(checkpoint, os.path.join(OUTPUT_WEIGHTS, fname))
            if args.model is None:
                simplex_model.add_vert()

    torch.save(reg_pars, os.path.join(OUTPUT_WEIGHTS, 'reg_params_' + CURRENT_TIME + '.pth'))
    logger.info("Params present in current model")
    for name, param in simplex_model.named_parameters():
        logger.info(name)
    torch.save(simplex_model.state_dict(), os.path.join(OUTPUT_WEIGHTS, 'state_dict_test_' + CURRENT_TIME + '.pth'))
    logger.info(utils.eval_avalanche(testloader, simplex_model, criterion))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--data_path",
        default='/tmp/datasets/',
        help="path to dataset",
    )

    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.05,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--simplex_lr",
        type=float,
        default=0.01,
        help="learning rate for training simplex",
    )
    parser.add_argument(
        "--LMBD",
        type=float,
        default=1e-6,
        metavar="lambda",
        help="value for \lambda in regularization penalty",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "--base_epochs",
        type=int,
        default=1,
        help="Number of epochs to train base model",
    )
    parser.add_argument(
        "--simplex_epochs",
        type=int,
        default=2,
        metavar="verts",
        help="Number of epochs to train additional simplex vertices",
    )
    parser.add_argument(
        "--n_component",
        type=int,
        default=1,
        help="total number of ensemble components",
    )

    parser.add_argument(
        "--n_verts",
        type=int,
        default=2,
        help="total number of vertices per simplex",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        help="number of samples to use per iteration",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        metavar="N",
        help="evaluate every n epochs",
    )
    parser.add_argument(
        "--task_number",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--model",
        type=str,
        default='output_weights/2_epochs_2_vertices.pt',
    )
    args = parser.parse_args()

    logger = setup_logger()
    main(args)
