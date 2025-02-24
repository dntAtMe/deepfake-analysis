"""Dataset class for loading raw audio files."""
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
import librosa


class RawAudioDataset(Dataset):
    """Dataset for loading raw audio files."""
    
    def __init__(
        self,
        metadata_path: Path,
        sample_rate: int = 16000,
        duration: Optional[float] = 4.0  # Duration in seconds
    ):
        """Initialize dataset from metadata file.
        
        Args:
            metadata_path: Path to metadata.pt file containing file paths and labels
            sample_rate: Target sample rate
            duration: Target duration in seconds (None for variable length)
        """
        data = torch.load(metadata_path)
        
        # Get wav directory from metadata path
        wav_dir = metadata_path.parent.parent / "wav"
        if not wav_dir.exists():
            raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
            
        # Map spectrogram paths to WAV paths
        self.wav_paths = [
            wav_dir / f"{Path(p).stem}.wav" 
            for p in data['files']
        ]
        
        # Verify WAV files exist
        missing_files = [p for p in self.wav_paths if not p.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing WAV files:\n" + 
                "\n".join(str(p) for p in missing_files[:5]) +
                f"\n...and {len(missing_files)-5} more" if len(missing_files) > 5 else ""
            )
        
        self.labels = data['labels']
        self.label_mapping = data['label_mapping']
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration) if duration else None
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load audio file and return with its label."""
        try:
            # Load audio with error handling
            audio, sr = librosa.load(
                self.wav_paths[idx],
                sr=self.sample_rate,
                duration=self.duration,
                mono=True
            )
            
            # Handle variable length or ensure fixed length
            if self.target_length:
                if len(audio) < self.target_length:
                    # Pad short audio
                    audio = librosa.util.fix_length(audio, size=self.target_length)
                elif len(audio) > self.target_length:
                    # Truncate long audio
                    audio = audio[:self.target_length]
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            return audio_tensor, self.labels[idx]
            
        except Exception as e:
            print(f"Error loading {self.wav_paths[idx]}: {str(e)}")
            raise
    
    def get_label_counts(self) -> Dict[str, int]:
        """Return count of samples per class."""
        counts = {}
        for label_idx, label_name in self.label_mapping.items():
            counts[label_name] = sum(1 for l in self.labels if l == label_idx)
        return counts
