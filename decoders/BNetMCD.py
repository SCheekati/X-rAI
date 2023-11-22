import torch
import torch.nn as nn
import torch.nn.functional as F

class BNetMCD(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, p_mc_dropout=0.5):
        super(BNetMCD, self).__init__()
        
        self.input_shape = input_shape
        self.flattened_size = input_shape[1] * input_shape[2]  # Flatten only the last two dimensions

        self.dropout_rate = p_mc_dropout

        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, data, apply_dropout=False):
        x = data[-1] 

        x = x.view(x.size(0), -1)  # shape will be [8, 1024*768]

        x = F.relu(self.fc1(x))
        if apply_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.fc2(x)
        if apply_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x
