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

    def __init__(self, bucket_name, npy_files, labels_dir, transform=None, stride=1):
        """
        Arguments:
            npy_file (string):  Path to npy file with data.
            root_dir (string):  Directory hosting all scans.
            transform (callable, optional): Optional transforms to be applied on a sample. 
        """
        self.ct_frames_list = [download_numpy_array(bucket_name, f) for f in npy_files]
        self.labels_map = self.load_labels_from_csv(labels_dir)
        self.transform = transform
        self.bucket_name = bucket_name
        self.npy_files = npy_files
        self.stride = stride

    def __len__(self):
        return len(self.ct_frames_list)
    
    def load_labels_from_csv(self, labels_csv):
        df = pd.read_csv(labels_csv)
        labels_map = dict(zip(df['idx'], df['label']))
        return labels_map
    
    def __getitem__(self, idx):
        ct_frames = self.ct_frames_list[idx]  # Get the frames for the current file

        # Assume ct_frames is a 3D array (num_slices, height, width)
        # For simplicity, let's take the first slice as the 2D frame
        if self.transform:
            ct_frames = self.transform(ct_frames)

        file_name = os.path.basename(self.npy_files[idx])
        label_index = self.get_label_index(file_name)
        label = self.labels_map[label_index]  # Assuming the label index corresponds to the file index

        sample = {"image": ct_frames, "label": label, "file_name": file_name}
        return sample


    def get_label_index(self, file_name):
        # Extract the part of the filename before '.npy' and convert to integer
        label_index = int(file_name.split('.')[0])
        return label_index
    
def custom_collate(batch):
    # Separate images, labels, and file names
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    # Convert numpy arrays to PyTorch tensors
    # Since the shapes can be different, we don't stack them
    images = [torch.from_numpy(img) for img in images]

    # Convert labels to a tensor
    labels = torch.tensor(labels)

    return {'image': images, 'label': labels, 'file_name': file_names}
    
ct_set = CTScanDataset(
    bucket_name="x_rai-dataset",
    npy_files=["pre_processed/multimodalpulmonaryembolismdataset/0/1004.npy", "pre_processed/multimodalpulmonaryembolismdataset/0/0.npy", "pre_processed/multimodalpulmonaryembolismdataset/0/1.npy", "pre_processed/multimodalpulmonaryembolismdataset/0/10.npy", "pre_processed/multimodalpulmonaryembolismdataset/0/100.npy", "pre_processed/multimodalpulmonaryembolismdataset/0/1001.npy",],
    labels_dir="data/Labels.csv",
    transform=None,
    stride = 1
)

trainloader = DataLoader(ct_set, batch_size=4,
                        shuffle=False, num_workers=0, collate_fn=custom_collate)
# Assuming you have already fetched the batch using the DataLoader
batch = next(iter(trainloader))

# Iterate through each item in the batch
# Iterate through each batch in the DataLoader
for batch_num, batch in enumerate(trainloader):
    print(f"Batch {batch_num + 1}")

    # Iterate through each item in the batch
    for i in range(len(batch['image'])):
        # Extract the 0th frame from the i-th item in the batch
        print(f"File name for CT Scan {i} in Batch {batch_num + 1}: {batch['file_name'][i]}")
        slice_to_display = batch['image'][i][0, :, :]  # 0th slice, all rows, all columns

        plt.imshow(slice_to_display.numpy(), cmap='gray')  # Display the slice
        plt.title(f"CT Scan {i} in Batch {batch_num + 1} - 0th Frame")
        plt.axis('off')
        plt.show()




