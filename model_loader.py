import sys

sys.path.append("../../simplex/")
sys.path.append("../../simplex/models/")

from torch.utils.data import DataLoader
from torchvision import transforms
from vgg_noBN import VGG16Simplex
from simplex_models import SimplexNet
from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
import torch
import utils

model = SimplexNet(10, VGG16Simplex, n_vert=2)
vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.load_state_dict(torch.load('state_dict_test3.pth'))

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

split_cifar_0 = SplitCIFAR10(n_experiences=5, shuffle=False, train_transform=transform_train,
                             eval_transform=transform_test)

print("Train streams")
for i in split_cifar_0.train_stream:
    print("Classes", i.classes_in_this_experience)
    print(len(i.dataset))

print("Test datasets")
for i in split_cifar_0.test_stream:
    print("Classes", i.classes_in_this_experience)
    print(len(i.dataset))

train_stream = split_cifar_0.train_stream
test_stream = split_cifar_0.test_stream

train_values = []
for i in train_stream:
    train_values.append(i)

test_values = []
for i in test_stream:
    test_values.append(i)

criterion = torch.nn.CrossEntropyLoss()

for index, i in enumerate(test_values):
    print(index, i.classes_in_this_experience)
    loader = DataLoader(i.dataset, shuffle=False)
    print(utils.eval_avalanche(loader, model, criterion))
    print(utils.eval_avalanche(loader, vgg_model, criterion))
