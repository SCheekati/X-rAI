import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import os


class CTScanDataset(Dataset):
    """CT Scan dataset."""

    def __init__(self, npy_file, labels_dir, transform=None):
        """
        Arguments:
            npy_file (string):  Path to npy file with data.
            root_dir (string):  Directory hosting all scans.
            transform (callable, optional): Optional transforms to be applied on a sample. 
        """
        frame = np.load(npy_file)[0:5, :, :]
        frame = np.reshape(frame, (1, frame.shape[0], frame.shape[1], frame.shape[2]))
        frame = torch.transpose(torch.from_numpy(frame), 1, 3)
        self.ct_frame = frame
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return self.ct_frame.size()[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.labels_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {"image": self.ct_frame, "label": self.labels_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample
