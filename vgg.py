import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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