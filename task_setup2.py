from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

from avalanche.benchmarks import SplitCIFAR10
from avalanche.training.strategies import AR1


def main():
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose(
        [Resize(224), ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_transform = transforms.Compose(
        [Resize(224), ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # ---------

    # --- SCENARIO CREATION
    scenario = SplitCIFAR10(
        5, train_transform=train_transform, eval_transform=test_transform
    )
    # ---------

    # CREATE THE STRATEGY INSTANCE
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )
    cl_strategy = Naive(model, optimizer, criterion, train_mb_size=100, train_epochs=1, eval_mb_size=100)

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience, num_workers=0)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=0))

main()