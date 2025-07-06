# dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

class CMUMOSEIDataset(Dataset):
    def __init__(self, data_path, split='train', absence_seed=0, modality_absence_rate=0.7):
        super().__init__()
        self.split = split
        self.absence_rate = modality_absence_rate
        self.absence_seed = absence_seed

        # Load dataset (each is a tensor or list of tensors)
        data = torch.load(os.path.join(data_path, f'{split}.pt'))
        self.vision = data['vision']
        self.audio = data['audio']
        self.text = data['text']
        self.labels = data['labels']

        # Seed for reproducibility
        random.seed(absence_seed)
        self.modality_masks = self.generate_modality_masks()

    def generate_modality_masks(self):
        masks = []
        for _ in range(len(self.labels)):
            # By default all modalities are present
            mask = {'V': 1, 'A': 1, 'T': 1}
            if random.random() < self.absence_rate:
                # Randomly drop one modality
                drop = random.choice(['V', 'A', 'T'])
                mask[drop] = 0
            masks.append(mask)
        return masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mask = self.modality_masks[idx]
        x_dict = {
            'V': self.vision[idx] if mask['V'] else None,
            'A': self.audio[idx] if mask['A'] else None,
            'T': self.text[idx] if mask['T'] else None
        }
        return {
            'inputs': x_dict,
            'modality_mask': mask,
            'labels': self.labels[idx]
        }

def get_dataloaders(config, absence_seed):
    """
    Returns train, val, test dataloaders for CMU-MOSEI.
    Args:
        config: dict with keys like data_path, batch_size, kappa
        absence_seed: used to seed absence pattern generation
    """
    data_path = config['data_path']
    batch_size = config.get('batch_size', 32)
    kappa = config.get('kappa', 1.0)
    modality_absence_rate = 1.0 if kappa == 1.0 else 0.7

    train_set = CMUMOSEIDataset(data_path, 'train', absence_seed, modality_absence_rate)
    val_set = CMUMOSEIDataset(data_path, 'val', absence_seed, modality_absence_rate)
    test_set = CMUMOSEIDataset(data_path, 'test', absence_seed, modality_absence_rate)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
