import collections
import numpy as np
import argparse
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
import numpy as np
import jaxlib
import pickle

def compare_models(model1, model2):
    matrix1 = model1.par_vectors().detach().numpy()
    matrix2 = model2.par_vectors().detach().numpy()

    # Extracts params and sorts it based on each dimension
    # print(matrix1)
    # print(matrix2)

    # Computes distance from ith point to jth point and stores it in valuedict
    valueDict = collections.defaultdict(lambda: collections.defaultdict(float))
    for i in range(len(matrix1)):
        for j in range(len(matrix2)):
            valueDict[i][j] = np.linalg.norm(matrix1[i] - matrix2[j])

    # print(valueDict)
    # Pairs based on euclidean distance between individual points
    alreadyPaired = set()
    matrix_sorted_1 = []
    matrix_sorted_2 = []
    for i in valueDict:
        # print(valueDict[i])
        sortedDist = []
        for key in valueDict[i]:
            sortedDist.append([key, valueDict[i][key]])
        sortedDist = sorted(sortedDist, key=lambda x:x[1])
        # print(sortedDist)
        # TODO: Find number of overlap, draw a heatmap (sort on one axis)
        # TODO: Find number of nodes close to it
        matrix_sorted_1.append(matrix1[i])
        matrix_sorted_2.append(matrix2[sortedDist[0][0]])
        # for j in sortedDist:
        #     if j[0] in alreadyPaired:
        #         continue
        #     else:
        #         alreadyPaired.add(j[0])
        #         matrix_sorted_1.append(matrix1[i])
        #         matrix_sorted_2.append(matrix2[j[0]])
        #         break

    # print("Sorted by distance")
    # print(matrix_sorted_1)
    # print(matrix_sorted_2)
    # Returns euclidean distance of the matrices
    total = 0.0
    for i in range(len(matrix_sorted_1)):
        total += pow(np.linalg.norm(matrix_sorted_1[i]-matrix_sorted_2[i]),2)

    return pow(total, 0.5)


if __name__ == '__main__':
    import os
    directories = os.listdir('output_weights_4')
    for i in range(len(directories)):
        for j in range(i+1, len(directories)):
            model1 = SimplexNet(2, SpiralModel, n_vert=5)
            model1.load_state_dict(torch.load('output_weights_4//' + directories[i] + '//base_2_simplex_4.pt'))
            model2 = SimplexNet(2, SpiralModel, n_vert=5)
            model1.load_state_dict(torch.load('output_weights_4//' + directories[j] + '//base_2_simplex_4.pt'))
            # print(directories[i], 'vs', directories[j])
            print(i,j, compare_models(model1, model2))

    for i in range(len(directories)):
        model1 = SimplexNet(2, SpiralModel, n_vert=5)
        model1.load_state_dict(torch.load('output_weights_4//' + directories[i] + '//base_2_simplex_4.pt'))
        print("Model", i)
        print("-"*80)
        print(model1.par_vectors().detach().numpy())
        print("-"*80)




