"""Dataset class for loading preprocessed spectrograms."""
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Dataset for loading preprocessed spectrograms."""
    
    def __init__(self, metadata_path: Path, device: str = "cpu"):
        """Initialize dataset from metadata file.
        
        Args:
            metadata_path: Path to metadata.pt file containing file paths and labels
            device: Device to load tensors onto ('cpu' or 'cuda')
        """
        data = torch.load(metadata_path, weights_only=True)
        self.file_paths = [Path(p) for p in data['files']]
        self.labels = data['labels']
        self.label_mapping = data['label_mapping']
        self.device = device
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load spectrogram and return with its label."""
        spec = torch.load(self.file_paths[idx], weights_only=True)
        label = self.labels[idx]
        return spec.to(self.device), label
    
    def get_label_counts(self) -> Dict[str, int]:
        """Return count of samples per class."""
        counts = {}
        for label_idx, label_name in self.label_mapping.items():
            counts[label_name] = sum(1 for l in self.labels if l == label_idx)
        return counts
