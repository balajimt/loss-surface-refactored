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

# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model = SimplexNet(10, VGG16Simplex, n_vert=3)
model.load_state_dict(torch.load('output_weights/07_05_2022_23_35_40/base_1_simplex_2.pt'))

model1 = SimplexNet(10, VGG16Simplex, n_vert=5)
model1.load_state_dict(torch.load('output_weights/13_05_2022_03_40_24/base_1_simplex_2.pt'))

model2 = SimplexNet(10, VGG16Simplex, n_vert=3)
model2.load_state_dict(torch.load('naive.pt'))

model3 = SimplexNet(10, VGG16Simplex, n_vert=3)
model3.load_state_dict(torch.load('2_replay.pt'))

compare_models(model1, model2)
compare_models(model1, model3)
compare_models(model2, model3)

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4
# )
#
# criterion = CrossEntropyLoss()
# cl_strategy = Replay(model, optimizer, criterion, train_mb_size=128, train_epochs=20, eval_mb_size=128)
#
# results = []
# test_stream = get_tasks(isTrain=False)
# train_stream = get_tasks(isTrain=True)
# results.append(cl_strategy.eval(test_stream))
# print(results)
#
# count = 0
# for experience in train_stream:
#     cl_strategy.train(experience)
#     cl_strategy.eval(test_stream)
#     count += 1
#     checkpoint = cl_strategy.model.state_dict()
#     torch.save(checkpoint, str(count)+"_replay.pt")

# # cl_strategy.train(train_stream)
# checkpoint = cl_strategy.model.state_dict()
# fname = "replay2.pt"
# torch.save(checkpoint, fname)

# for experience in train_stream:
#     print("Start of experience: ", experience.current_experience)
#     print("Current Classes: ", experience.classes_in_this_experience)
#     # count += 1
#     # if count != 1:
#     cl_strategy.train(experience)
#     cl_strategy.eval(test_stream)
#     # print('Training completed')
#
#     print('Computing accuracy on the whole test set')
# results.append(cl_strategy.eval(test_stream))

# split_cifar = SplitCIFAR10(n_experiences=5, fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], return_task_id=True,
#                            shuffle=False)
# train_stream = split_cifar.train_stream
# test_stream = split_cifar.test_stream
#
# for exp in train_stream:
#     t = exp.task_label
#     exp_id = exp.current_experience
#     task_train_ds = exp.dataset
#     print('Task {} batch {} -> train'.format(t, exp_id))
#     print('This batch contains', len(task_train_ds), 'patterns')
#
# for exp in test_stream:
#     t = exp.task_label
#     exp_id = exp.current_experience
#     task_train_ds = exp.dataset
#     print('Task {} batch {} -> test'.format(t, exp_id))
#     print('This batch contains', len(task_train_ds), 'patterns')
#
# # model = SimpleMLP(num_classes=10)
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4
# )
# # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# criterion = CrossEntropyLoss()
# cl_strategy = Naive(model, optimizer, criterion,train_mb_size=100, train_epochs=2, eval_mb_size=100)
#
# results = []
# # for experience in train_stream:
# #     print("Start of experience: ", experience.current_experience)
# #     print("Current Classes: ", experience.classes_in_this_experience)
# #
# #     # cl_strategy.train(experience)
# #     # print('Training completed')
# #
# #     print('Computing accuracy on the whole test set')
# results.append(cl_strategy.eval(test_stream))
#
# print(results)
