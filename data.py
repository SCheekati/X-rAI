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

def window_ct_scan(ct_frames, window_size):
    """
    Slices a 3D CT scan into windows of a specified size.

    Parameters:
    ct_frames (numpy array): The 3D array representing the CT scan.
    window_size (int): The number of slices in each window.

    Returns:
    list of numpy arrays: A list where each element is a window of the CT scan.
    """
    # windows = []
    # for start in range(len(ct_frames) // 2 - 17, len(ct_frames) // 2 + 19):
    #     end = start + window_size
    #     window = ct_frames[start:end]
    #     windows.append(window)
    # windows = torch.stack(windows, dim=0)
    #return windows
    num_slices = ct_frames.shape[0]
    desired_window_size = 40

    if num_slices >= desired_window_size:
        windows = []
        for start in range(len(ct_frames) // 2 - 20, len(ct_frames) // 2 + 16):
            end = start + window_size
            window = ct_frames[start:end]
            windows.append(window)
        windows = torch.stack(windows, dim=0)
        return windows
    else:
        # If not enough slices, duplicate slices in order
        duplicated_slices = ct_frames
        temp = []
        while len(duplicated_slices) < desired_window_size:
            for i in range(num_slices):
                temp.append(duplicated_slices[i])
                temp.append(duplicated_slices[i])
            num_slices *= 2
            duplicated_slices = temp
        return window_ct_scan(torch.stack(duplicated_slices, dim=0), window_size)

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    return [blob.name for blob in blobs]
class CTScanDataset(Dataset):
    """CT Scan dataset."""

    def __init__(self, bucket_name, npy_files, labels_dir, transform=None, stride=1, windowing_enabled=False):
        """
        Arguments:
            npy_files (list):  list to npy file with data.
            root_dir (string):  Directory hosting all scans.
            transform (callable, optional): Optional transforms to be applied on a sample. 
            windowing_enabled (bool): Whether to enable windowing of CT scans.
        """
        self.labels_map = self.load_labels_from_csv(labels_dir)
        self.transform = transform
        self.bucket_name = bucket_name
        self.npy_files = npy_files
        self.stride = stride
        self.windowing_enabled = windowing_enabled

    def __len__(self):
        return len(self.npy_files)
    
    def load_labels_from_csv(self, labels_csv):
        df = pd.read_csv(labels_csv)
        labels_map = dict(zip(df['idx'], df['label']))
        return labels_map
    
    def __getitem__(self, idx):
        ct_frames = torch.from_numpy(download_numpy_array(self.bucket_name, self.npy_files[idx]))  # Get the frames for the current file
        windows = window_ct_scan(ct_frames, 5)  # Not windowing, use the entire scan as one item
        #print(windows.shape)
        # 403 x 5 x 224 x 224
        windows = torch.reshape(windows, (1, windows.shape[0], 1, windows.shape[2], windows.shape[3], windows.shape[1]))
        if self.transform:
            windows = [self.transform(ct_frames)]

        file_name = os.path.basename(self.npy_files[idx])
        label_index = self.get_label_index(file_name)
        label = self.labels_map[label_index]

        samples = [{"image": window, "label": label, "file_name": file_name} for window in windows]
        return samples


    def get_label_index(self, file_name):
        # Extract the part of the filename before '.npy' and convert to integer
        label_index = int(file_name.split('.')[0])
        return label_index
    
def custom_collate(batch):
    # Initialize empty lists to hold images, labels, and file names
    images = []
    labels = []
    file_names = []

    for item in batch:
        for sub_item in item:
            # Extract images, labels, and file names from each sub-item
            images.append(sub_item['image'])
            labels.append(sub_item['label'])
            file_names.append(sub_item['file_name'])

    # Convert numpy arrays to PyTorch tensors
    #images = [torch.from_numpy(img) for img in images]

    # Convert labels to a tensor
    labels = torch.tensor(labels)

    return {'image': images, 'label': labels, 'file_name': file_names}


def main():
    bucket_name = "x_rai-dataset"
    prefix = "resized/pre_processed/multimodalpulmonaryembolismdataset/" 
    ct_set = CTScanDataset(
        bucket_name="x_rai-dataset",
        npy_files=list_blobs_with_prefix(bucket_name, prefix),
        labels_dir="data/Labels.csv",
        transform=None,
        stride = 5
    )

    trainloader = DataLoader(ct_set, batch_size=2,
                            shuffle=False, num_workers=3, collate_fn=custom_collate)
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
            
if __name__ == '__main__':
    main()



