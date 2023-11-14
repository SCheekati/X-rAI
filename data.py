import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
import io
import matplotlib.pyplot as plt
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "x-rai-403303-d80059f325d8.json"

def download_numpy_array(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    byte_stream = io.BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)

    return np.load(byte_stream, allow_pickle=True)
class CTScanDataset(Dataset):
    """CT Scan dataset."""

    def __init__(self, bucket_name, npy_file, labels_dir, transform=None, stride=1):
        """
        Arguments:
            npy_file (string):  Path to npy file with data.
            root_dir (string):  Directory hosting all scans.
            transform (callable, optional): Optional transforms to be applied on a sample. 
        """
        npy_array = download_numpy_array(bucket_name, npy_file)
        file_name = os.path.basename(npy_file)

        # Remove the '.npy' extension to get the identifier (e.g., '0')
        self.ct_idx = int(file_name.split('.')[0])
        self.ct_frames = npy_array
        self.labels_map = self.load_labels_from_csv(labels_dir)
        self.transform = transform
        self.bucket_name = bucket_name
        self.npy_file = npy_file
        self.stride = stride

    def __len__(self):
        return (self.ct_frames.shape[0] - 5) // self.stride + 1
    
    def load_labels_from_csv(self, labels_csv):
        df = pd.read_csv(labels_csv)
        labels_map = dict(zip(df['idx'], df['label']))
        return labels_map
    
    def __getitem__(self, idx):
        start_frame = idx * self.stride
        end_frame = start_frame + 5
        
        frame_slice = self.ct_frames[start_frame:end_frame, :, :]
        frame = np.reshape(frame_slice, (1, frame_slice.shape[0], frame_slice.shape[1], frame_slice.shape[2]))
        frame = torch.transpose(torch.from_numpy(frame), 1, 3)
        
        label = self.labels_map[self.ct_idx]
        print(f"Label for scan {self.ct_idx}: {label}")
        
        sample = {"image": frame, "label": label}
        if self.transform:
            sample = self.transform(sample)
            
        print(f"Frame range for scan {self.ct_idx}: {start_frame} to {end_frame - 1}")
        return sample
    
ct_set = CTScanDataset(
    bucket_name="x_rai-dataset",
    npy_file="pre_processed/multimodalpulmonaryembolismdataset/0/1004.npy",
    labels_dir="data/Labels.csv",
    transform=None,
    stride = 1
)

trainloader = DataLoader(ct_set, batch_size=4,
                        shuffle=True, num_workers=0)
batch = next(iter(trainloader)) # Get a batch from the DataLoader
image_tensor = batch['image']
slice_to_display = image_tensor[3, 0, :, :, 1] # 4th item in batch, 0th channel, all rows, all columns, 2nd frame
plt.imshow(slice_to_display.numpy(), cmap='gray') # Display the slice
plt.axis('off')
plt.show()

