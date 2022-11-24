import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os.path
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import gpytorch
import copy
import matplotlib as mpl
import cmocean
import cmocean.cm as cmo
from matplotlib import colors


import sys
sys.path.append("../simplex/")
import utils
from simplex_helpers import volume_loss, complex_volume
import surfaces

sys.path.append("../simplex/models/")
from basic_mlps import BasicNet, BasicSimplex
from simplex_models import SimplexNet, Simplex
from preresnet import PreResNetSimplex

def compute_loss_surface(model, train_x, train_y, v1, v2,
                         loss, n_pts=50, range_=10.):

    start_pars = model.state_dict()
    vec_lenx = torch.linspace(-range_.item(), range_.item(), n_pts)
    vec_leny = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_lenx[ii]) + v2.mul(vec_leny[jj])
                # print(perturb.shape)
                perturb = utils.unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)

                loss_surf[ii, jj] = loss(model(train_x), train_y)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_lenx, vec_leny)
    return X, Y, loss_surf

def twospirals(n_points, noise=.3, random_state=920):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 600 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

noise = 1.

X, Y = twospirals(50, noise=noise)
train_x = torch.FloatTensor(X)
train_y = torch.FloatTensor(Y).unsqueeze(-1)

X, Y = twospirals(100, noise=noise)
test_x = torch.FloatTensor(X)
test_y = torch.FloatTensor(Y).unsqueeze(-1)

# train_x, train_y = train_x.cuda(), train_y.cuda()
# test_x, test_y = test_x.cuda(), test_y.cuda()

def trainer(model, train_x, train_y, reg_par=1e-5, niters=500,
            print_every=100, print_=False, n_samples=5):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for step in range(niters):
        optimizer.zero_grad()
        vol = model.total_volume() + 1e-4
        #         print(vol)
        acc_loss = 0
        for ii in range(n_samples):
            outputs = model(train_x)
            acc_loss += loss_func(outputs, train_y)
        loss = acc_loss - reg_par * vol.log()
        if print_:
            if step % print_every == 0:
                print(acc_loss.item())
                print(vol.item())
                print("\n")

        loss.backward()
        optimizer.step()

LMBD = 3.

reg_pars = [0.]
for ii in range(2, 5):
    fix_pts = [True]*(ii)
    start_vert = len(fix_pts)

    out_dim = 1
    temp = SimplexNet(out_dim, BasicSimplex, n_vert=start_vert,
                      fix_points=fix_pts,)
    log_vol = (temp.total_volume() + 1e-4).log()
    reg_pars.append(max(float(LMBD)/log_vol, 1e-8).item())

print(reg_pars)

fix_pts = [False]
start_vert = len(fix_pts)

total_verts = 4

out_dim = 1
arch_kwargs = {"in_dim":2, "hidden_size":25,
               "activation":torch.nn.ReLU(), "bias":True}

simplex_model = SimplexNet(out_dim, BasicSimplex, n_vert=start_vert,
                           fix_points=fix_pts,
                           architecture_kwargs=arch_kwargs)
simplex_model = simplex_model

base_model = BasicNet(simplex_model.n_output, **simplex_model.architecture_kwargs)
base_model = base_model

trainer(simplex_model, train_x, train_y, reg_par=reg_pars[0], print_=True, n_samples=1, niters=200)

for ii in range(1, total_verts):
    simplex_model.add_vert()
    simplex_model = simplex_model
    trainer(simplex_model, train_x, train_y, reg_par=0., print_=True, niters=100)