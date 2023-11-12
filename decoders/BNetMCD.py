import torch
import torch.nn as nn
import torch.nn.functional as F

class BNetMCD(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, p_mc_dropout=0.5):
        super(BNetMCD, self).__init__()
        
        self.input_shape = input_shape
        self.flattened_size = input_shape[0] * input_shape[1] * input_shape[2] # gonna be crazy?
        self.dropout_rate = p_mc_dropout

        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, apply_dropout=False):
        x = x.view(-1, self.flattened_size) #  flatten

        x = F.relu(self.fc1(x))
        if apply_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.fc2(x)
        if apply_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return x
