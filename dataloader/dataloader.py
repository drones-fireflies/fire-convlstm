import os
import torch
import numpy as np

from torch.utils.data import Dataset

# ================
#  PyTorch Dataset
# ================
class PyTorchWildfireDataset(Dataset):
    """
        Args:
            file_path (str): Path to the `.npz` file.
            history_length (int): Number of past timesteps used as input.
            output_channels: Subset of channels to predict.
    """
    
    def __init__(self, file_path: str, history_len=1, output_channels=slice(0, 2)):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
        data = np.load(file_path)
        
        self.data = data['X']
        
        self.output_channels = output_channels
        self.history_len = history_len

        self.samples = []
        num_samples, seq_length, _, _, _ = self.data.shape

        # Build samples
        self.samples = [
            (sample_idx, timestep)
            for sample_idx in range(num_samples)
            for timestep in range(seq_length - self.history_len)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, t = self.samples[idx]
       
        input = self.data[i, t:t+self.history_len] 
        target = self.data[i, t+self.history_len, self.output_channels, :, :] 
        
        return torch.from_numpy(input).float(), torch.from_numpy(target).float()
