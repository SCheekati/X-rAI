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

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

bucket_name = "x_rai-dataset"
prefix = "resized/pre_processed/multimodalpulmonaryembolismdataset/" 
npy_files = list_blobs_with_prefix(bucket_name, prefix)
minshape = 1000000
for i in range(len(npy_files)):
    ct_frames = torch.from_numpy(download_numpy_array(bucket_name, npy_files[i])) 
    print(ct_frames.shape)
    minshape = min(minshape, ct_frames.shape[0])
print("min: ", minshape)
