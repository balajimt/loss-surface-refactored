import sys

sys.path.append("../../simplex/")
sys.path.append("../../simplex/models/")

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
from vgg_noBN import VGG16Simplex
import numpy as np
import jaxlib
import pickle
import task_splitter

# GPU specifics
torch.cuda.set_device(0)
print("GPU", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU NAME", torch.cuda.get_device_name(0))


spiral_dataset = True

def make_spirals(n_samples, noise_std=0., rotations=1.):
    ts = jnp.linspace(0, 1, n_samples)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * np.pi
    signs = np.random.randint(0, 2, (n_samples,)) * 2 - 1
    labels = (signs > 0).astype(int)

    xs = rs * signs * jnp.cos(thetas) + np.random.randn(n_samples) * noise_std
    ys = rs * signs * jnp.sin(thetas) + np.random.randn(n_samples) * noise_std
    points = jnp.stack([xs, ys], axis=1)
    return points, labels

def create_torch_dataset(x_tensors, y_tensors):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    tensor_x = torch.Tensor(x_tensors) # transform to torch tensor
    tensor_y = torch.Tensor(y_tensors).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    return DataLoader(my_dataset) # create your dataloader



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
    logger.info(str(["LOGGING TO", os.path.abspath(filename)]))
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def main(args):

    task_number = 0
    trainloader = task_splitter.get_task_dataloader(task_number, True)
    testloader = task_splitter.get_task_dataloader(task_number, False)
    logger.info(str(["Successfully loaded train and test streams from task number", task_number]))
    try:
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
            global simplex_model
            savedir = "./saved-outputs/"

            reg_pars = []
            ## randomly initialize simplexes to determine regularization parameters ##
            for ii in range(args.n_verts + 1):
                fix_pts = [True] * (ii + 1)
                start_vert = len(fix_pts)

                out_dim = 2
                simplex_model = SimplexNet(out_dim, SpiralModel, n_vert=start_vert,
                                           fix_points=fix_pts)
                simplex_model = simplex_model.cuda()

                log_vol = (simplex_model.total_volume() + 1e-4).log()

                reg_pars.append(max(float(args.LMBD) / log_vol, 1e-8))



            # for ii in range(initial + 1):
            #     fix_pts = [True] * (ii + 1)
            #     start_vert = len(fix_pts)
            #
            #     out_dim = 2
            #     simplex_model = SimplexNet(out_dim, SpiralModel, n_vert=start_vert,
            #                                fix_points=fix_pts)
            #     simplex_model = simplex_model.cuda()
            #
            #     log_vol = (simplex_model.total_volume() + 1e-4).log()
            #
            #     reg_pars.append(max(float(args.LMBD) / log_vol, 1e-8))

            # Uncomment for spiral dataset
            # ## import training and testing data ##
            #
            # train_frac = 0.8
            #
            # ## Two spirals dataset
            # train_points, train_labels = make_spirals(int(train_frac * 1000), noise_std=0.05)
            # with open(os.path.join(OUTPUT_WEIGHTS, 'train_points.pkl'), 'wb') as f:
            #     pickle.dump(train_points, f)
            # with open(os.path.join(OUTPUT_WEIGHTS, 'train_labels.pkl'), 'wb') as f:
            #     pickle.dump(train_labels, f)
            #
            # test_points, test_labels = make_spirals(int((1 - train_frac) * 1000), noise_std=0.05)
            # with open(os.path.join(OUTPUT_WEIGHTS, 'test_points.pkl'), 'wb') as f:
            #     pickle.dump(test_points, f)
            # with open(os.path.join(OUTPUT_WEIGHTS, 'test_labels.pkl'), 'wb') as f:
            #     pickle.dump(test_labels, f)
            #
            # trainloader = create_torch_dataset(np.array(train_points), np.array(train_labels))
            # testloader = create_torch_dataset(np.array(test_points), np.array(test_labels))
            # logger.info(str(["Successfully loaded train and test streams spiral dataset", len(train_points)]))

            # Simplex code starts here
            columns = ['component', 'vert', 'ep', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', "vol"]
            criterion = torch.nn.CrossEntropyLoss()
            for component in range(args.n_component):
                fix_pts = [False]
                simplex_model = SimplexNet(10, VGG16Simplex, n_vert=1, fix_points=fix_pts).cuda()
                # simplex_model.load_state_dict(torch.load('output_weights/07_05_2022_23_35_40/base_1_simplex_2.pt'))
                # simplex_model = SimplexNet(10, VGG16Simplex, n_vert=1,
                #                            fix_points=fix_pts)
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
                    logger.info("Saving to " + str(os.path.join(OUTPUT_WEIGHTS, fname)))
                    torch.save(checkpoint, os.path.join(OUTPUT_WEIGHTS, fname))
                    simplex_model.add_vert()

            torch.save(reg_pars, os.path.join(OUTPUT_WEIGHTS, 'reg_params_' + CURRENT_TIME + '.pth'))
            logger.info("Params present in current model")
            for name, param in simplex_model.named_parameters():
                logger.info(name)
            logger.info("Saving simplex model state dict to " + str(os.path.join(OUTPUT_WEIGHTS, 'state_dict_test_' + CURRENT_TIME + '.pth')))
            torch.save(simplex_model.state_dict(), os.path.join(OUTPUT_WEIGHTS, 'state_dict_test_' + CURRENT_TIME + '.pth'))
            # logger.info(utils.eval(testloader, simplex_model, criterion))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Spiral dataset simplex")

    parser.add_argument(
        "--data_path",
        default='/tmp/datasets/',
        help="path to dataset",
    )

    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="Task number",
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
        default=3,
        metavar="verts",
        help="Number of epochs to train additional simplex vertices",
    )
    parser.add_argument(
        "--n_component",
        type=int,
        default=3,
        help="total number of ensemble components",
    )

    parser.add_argument(
        "--n_verts",
        type=int,
        default=3,
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
    args = parser.parse_args()

    logger = setup_logger()
    main(args)
