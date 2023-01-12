import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from time import time
print('here')

start = time()

# Hyperparameters
IMG_SIZE = 224
IMG_CHANNELS = 3
BATCH_SIZE = 256
INIT_LR = 1e-2
EPOCHS = 74
MOMENTUM = 0.9
L2_PENALTY = 5e-4
MEAN = 0
VAR = 1e-2
NUM_CLASSES = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG(nn.Module):
    def __init__(self, layers, num_classes, img_channels):
        super(VGG, self).__init__()
        self.layers = layers
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.curr_in_features = self.img_channels

        self.conv_net = self.build_conv_model()
        self.linear_net = self.build_linear_model()
        self.init_weights()

    def forward(self, x):
        x = self.conv_net(x)
        x = self.linear_net(x)
        return x

    def init_weights(self):
        for layer in self.conv_net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, MEAN, np.sqrt(VAR))
                nn.init.constant_(layer.bias, 0)
        for layer in self.linear_net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, MEAN, np.sqrt(VAR))
                nn.init.constant_(layer.bias, 0)

    def build_conv_model(self):
        self.conv_layer = 1
        self.mp_layer = 1
        self.lrn_layer = 1
        self.relu_layer = 1
        model_ = nn.Sequential()
        for layer in self.layers:
            if isinstance(layer, int):
                model_.add_module(f'conv_{str(self.conv_layer)}', nn.Conv2d(self.curr_in_features, layer, 3, 1, 1))
                model_.add_module(f'relu_{str(self.relu_layer)}', nn.ReLU())
                self.curr_in_features = layer
                self.conv_layer += 1
                self.relu_layer += 1
            elif layer == 'lrn':
                model_.add_module(f'lrn_{str(self.lrn_layer)}', nn.LocalResponseNorm(self.curr_in_features))
                self.lrn_layer += 1
            elif layer == 'mp':
                model_.add_module(f'mp_{str(self.mp_layer)}', nn.MaxPool2d(2, 2))
                self.mp_layer += 1
        model_.add_module('flatten', nn.Flatten())

        return model_

    def build_linear_model(self):
        return nn.Sequential(
            # (N, 25088)
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # (N, 4096)
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # (N, 4096)
            nn.Linear(4096, self.num_classes),
            # (N, 1000)
            nn.Softmax(dim=1)
        )

vgg_11 = [64, 'mp', 128, 'mp', 256, 256, 'mp', 512, 512, 'mp', 512, 512, 'mp']
vgg_11_lrn = [64, 'lrn', 'mp', 128, 'mp', 256, 256, 'mp', 512, 512, 'mp', 512, 512, 'mp']
vgg_13 = [64, 64, 'mp', 128, 128, 'mp', 256, 256, 'mp', 512, 512, 'mp', 512, 512, 'mp']
vgg_16 = [64, 64, 'mp', 128, 128, 'mp', 256, 256, 256, 'mp', 512, 512, 512, 'mp', 512, 512, 512, 'mp']
vgg_19 = [64, 64, 'mp', 128, 128, 'mp', 256, 256, 256, 256, 'mp', 512, 512, 512, 512, 'mp', 512, 512, 512, 512, 'mp']

model_vgg_16 = VGG(vgg_16, NUM_CLASSES, IMG_CHANNELS)