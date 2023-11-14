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
        
        frame = np.load(npy_file)
        ct_frames = []
        for i in range(4):
            if i < 508:
                temp = frame[i:i+5, :, :]
            else:
                temp = frame[i:-1, :, :]
            temp = np.reshape(temp, (1, temp.shape[0], temp.shape[1], temp.shape[2]))
            temp = torch.transpose(torch.from_numpy(temp), 1, 3)
            ct_frames.append(temp.float())
        self.ct_frame = [ct_frames, ct_frames, ct_frames, ct_frames, ct_frames, ct_frames, ct_frames]
        labels_dir = labels_dir.float()
        self.labels_dir = [labels_dir, labels_dir, labels_dir, labels_dir, labels_dir, labels_dir, labels_dir]
        self.transform = transform

    def __len__(self):
        return len(self.ct_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.labels_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {"image": self.ct_frame[idx], "label": self.labels_dir[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
