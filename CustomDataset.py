import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, features, targets, transforms=None):
        self.features = features
        self.targets  = targets
        self.transforms = transforms

    def __len__(self):
        # Number of samples
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features_sample = self.features[idx]
        targets_sample  = self.targets[idx]

        if self.transforms:
            features_sample = self.transforms(features_sample)
            targets_sample  = self.transforms(targets_sample)

        sample = (features_sample, targets_sample)
    
        return sample
