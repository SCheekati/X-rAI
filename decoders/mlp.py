import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MlpHead(nn.Module):

    def __init__(
        self,
        channels=768,
        num_classes=14,
        dropout_ratio=0.1
    ):
      super().__init__()
      self.channels = channels
      self.num_classes = num_classes
      self.dropout_ratio = dropout_ratio

      self.fc1 = nn.Linear(channels, 64)
      self.d1 = nn.Dropout(p=dropout_ratio)
      self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
      x = x[-1]
      x = x.mean(dim=1)
      x = self.fc1(x)
      x = self.d1(x)
      x = self.fc2(x)
      #x = F.softmax(self.fc2(x), dim=1)
      return x