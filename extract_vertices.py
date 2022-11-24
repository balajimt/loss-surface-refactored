import os
import argparse
import sys

sys.path.append("../../simplex/")
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16Simplex
from simplex_models import SimplexNet
from vgg_noBN import SpiralModel

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


def get_input_files(input_directory):
    input_files = []
    for i in os.listdir(input_directory):
        if i.endswith('.pt'):
            input_files.append(os.path.join(input_directory, i))
    return input_files

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

    print(sum((x - y).abs().sum().pow(2) for x, y in zip(dict1.values(), dict2.values())).pow(0.5))

def extract_vertex_weights_for_spiral(input_directory):
    input_files = get_input_files(input_directory)
    output_directory = os.path.join(input_directory, 'vertex_weights/')
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print(input_files)

    for filename in input_files:
        if filename.endswith('.pt'):
            print(filename)
            name = filename.split("/")[-1]
            vertexes = int(name.split('_')[-1].split('.')[0])+1
            model = SimplexNet(2, SpiralModel, n_vert=vertexes)
            model.load_state_dict(torch.load(filename))
            vertex_weights = model.par_vectors()
            print(len(vertex_weights))
            print(len(vertex_weights[0]))
            with open(os.path.join(output_directory, name), 'wb') as f:
                pickle.dump(vertex_weights, f)

def extract_vertex_weights(input_directory):
    input_files = get_input_files(input_directory)
    output_directory = os.path.join(input_directory, 'vertex_weights/')
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print(input_files)

    for filename in input_files:
        print(filename)
        name = filename.split("/")[-1]
        vertexes = int(name.split('_')[-1].split('.')[0])+1
        model = SimplexNet(10, VGG16Simplex, n_vert=vertexes)
        model.load_state_dict(torch.load(filename))
        vertex_weights = model.par_vectors()
        print(len(vertex_weights))
        print(len(vertex_weights[0]))
        with open(os.path.join(output_directory, name), 'wb') as f:
            pickle.dump(vertex_weights, f)

def extract_naive_weights(n_vertices, input_directory):
    input_files = get_input_files(input_directory)
    output_directory = os.path.join(input_directory, 'vertex_weights/')
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print(input_files)

    for filename in input_files:
        print(filename)
        name = filename.split("/")[-1]
        model = SimplexNet(10, VGG16Simplex, n_vert=n_vertices)
        model.load_state_dict(torch.load(filename))
        vertex_weights = model.par_vectors()
        print(len(vertex_weights))
        print(len(vertex_weights[0]))
        with open(os.path.join(output_directory, name), 'wb') as f:
            pickle.dump(vertex_weights, f)

def extract_vertex_weights_plus5(input_directory):
    input_files = get_input_files(input_directory)
    output_directory = os.path.join(input_directory, 'vertex_weights/')
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print(input_files)

    for filename in input_files:
        print(filename)
        name = filename.split("/")[-1]
        vertexes = int(name.split('_')[-1].split('.')[0])+5
        model = SimplexNet(10, VGG16Simplex, n_vert=vertexes)
        model.load_state_dict(torch.load(filename))
        vertex_weights = model.par_vectors()
        print(len(vertex_weights))
        print(len(vertex_weights[0]))
        with open(os.path.join(output_directory, name), 'wb') as f:
            pickle.dump(vertex_weights, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar10 simplex")
    parser.add_argument(
        "--state_dict_folder",
        default='/tmp/loss-surface-refactored/naive/30_05_2022_11_58_46',
        help="Path to saved state dict from task 1",
    )
    parser.add_argument(
        "--n_vertices",
        type=int,
        default=5,
        help="Number of vertices of model",
    )

    args = parser.parse_args()

    # extract_vertex_weights_plus5('output_weights/29_05_2022_02_35_52')
    # extract_naive_weights(5, 'replay/29_05_2022_02_33_14') #75
    # extract_naive_weights(5, 'replay/29_05_2022_02_33_58') #50
    extract_vertex_weights_for_spiral('output_weights/24_07_2022_18_07_03/') #10