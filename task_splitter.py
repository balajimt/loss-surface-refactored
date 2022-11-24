import sys

import torch

sys.path.append("../../simplex/")
sys.path.append("../../simplex/models/")

import utils
import collections
from torchvision import transforms
from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
from torch.utils.data import DataLoader, ConcatDataset
import itertools

# Train and test transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Generates 5 tasks with the following classes at run time:
# [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
split_cifar = SplitCIFAR10(n_experiences=5, shuffle=False, train_transform=transform_train,
                           eval_transform=transform_test)

train_stream = split_cifar.train_stream
test_stream = split_cifar.test_stream

train_values = []
for i in train_stream:
    train_values.append(i)

test_values = []
for i in test_stream:
    test_values.append(i)


def get_tasks(isTrain):
    if isTrain:
        return train_stream
    return test_stream


def get_task_experience(index:int, isTrain: bool):
    if isTrain:
        return train_values[index]
    return test_values[index]


def get_task_dataloader(index: int, isTrain: bool):
    """
    Returns a specific avalanche task
    :param index: index of the task list (also known as task number)
    :param isTrain: boolean variable for training or test datasets
    :return:
    """
    if isTrain:
        return DataLoader(train_values[index].dataset, shuffle=False, batch_size=128)
    else:
        return DataLoader(test_values[index].dataset, shuffle=False, batch_size=128)


def get_2task_dataloader(isTrain: bool):
    """
    Returns a specific avalanche task
    :param index: index of the task list (also known as task number)
    :param isTrain: boolean variable for training or test datasets
    :return:
    """

    if isTrain:
        dataset_final = ConcatDataset([train_values[0].dataset, train_values[1].dataset])
        return DataLoader(dataset_final, shuffle=True, batch_size=128)
    else:
        dataset_final = ConcatDataset([test_values[0].dataset, test_values[1].dataset])
        return DataLoader(dataset_final, shuffle=True, batch_size=128)


def compare_models(model1, model2):
    """
    Function to compare two models - it first compares model performance on the same task
    and then compares the individual weight values
    :param model1: Model 1
    :param model2: Model 2
    """
    testloader = get_task_dataloader(0, False)
    criterion = torch.nn.CrossEntropyLoss()

    # print(utils.eval(testloader, model1, criterion))
    # print(utils.eval(testloader, model2, criterion))
    dict1 = collections.defaultdict(float)
    dict2 = collections.defaultdict(float)
    for name, param in model1.named_parameters():
        new_name = name.split("_")[0]
        dict1[new_name] = param
    for name, param in model2.named_parameters():
        new_name = name.split("_")[0]
        dict2[new_name] = param

    # diff = []
    # for key in dict1:
    #     abs(dict1[key] - dict2[key]).sum()
    # print(torch.cdist(torch.tensor(list(dict1.values())), torch.tensor(list(dict2.values()))))

    # for x, y in zip(dict1.values(), dict2.values()):
    #     print((x-y).sum().pow(2))

    print(sum((x - y).abs().sum().pow(2) for x, y in zip(dict1.values(), dict2.values())).pow(0.5))
