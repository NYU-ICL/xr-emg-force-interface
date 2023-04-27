from torch.utils.data import Dataset
import numpy as np
import torch
import os


class EMGDataset(Dataset):

    def __init__(self, root_dir):
        super(EMGDataset, self).__init__()
        self.emg = np.load(os.path.join(root_dir, 'emg_train.npy'))
        self.force = np.load(os.path.join(root_dir, 'force_train.npy'))
        self.force_class = np.load(os.path.join(root_dir, 'force_class_train.npy'))

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        emg = self.emg[idx]
        force = self.force[idx]
        force_class = self.force_class[idx]
        return torch.from_numpy(emg), torch.from_numpy(force), torch.from_numpy(force_class)
