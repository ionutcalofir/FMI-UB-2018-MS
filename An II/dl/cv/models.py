import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def resnext101_32x8d():
  return models.resnext101_32x8d(pretrained=True)
