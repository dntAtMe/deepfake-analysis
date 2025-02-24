"""Audio feature extraction utilities."""
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import librosa
import torch
from torch import Tensor
import scipy.fftpack

class FeatureType(Enum):
    """Supported feature types."""
    MFCC = "mfcc"
    LFCC = "lfcc"
    MEL = "mel"

class FeatureExtractor:
    """Audio feature extractor supporting multiple feature types."""
    
    def __init__(
        self,
        feature_type: Union[str, FeatureType] = "mfcc",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,  # 10ms at 16kHz
        n_mels: int = 80,
        n_mfcc: int = 40,
        n_lfcc: int = 40,
        fmin: int = 20,
        fmax: int = 8000,
        normalize: bool = True
    ):
        """Initialize feature extractor.
        
        Args:
            feature_type: Type of features to extract ("mfcc", "lfcc", or "mel")
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
            n_lfcc: Number of LFCC coefficients
            fmin: Minimum frequency
            fmax: Maximum frequency
            normalize: Whether to normalize features
        """
        self.feature_type = FeatureType(feature_type)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        
        # Pre-compute filters
        if self.feature_type in [FeatureType.MFCC, FeatureType.MEL]:
            self.mel_basis = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax
            )
        
        if self.feature_type == FeatureType.LFCC:
            # Create linear filterbank manually
            self.linear_basis = self._create_linear_filterbank(
                n_freqs=n_fft // 2 + 1,
                n_filter=n_lfcc,
                min_freq=fmin,
                max_freq=fmax,
                sample_rate=sample_rate
            )
    
    def _create_linear_filterbank(
        self,
        n_freqs: int,
        n_filter: int,
        min_freq: float,
        max_freq: float,
        sample_rate: int
    ) -> np.ndarray:
        """Create linear-spaced filterbank."""
        # Convert Hz to FFT bins
        min_bin = int(min_freq * n_freqs / (sample_rate / 2))
        max_bin = int(max_freq * n_freqs / (sample_rate / 2))
        
        # Create linearly spaced filters
        freq_points = np.linspace(min_bin, max_bin, n_filter + 2)
        filters = np.zeros((n_filter, n_freqs))
        
        # Create triangular filters
        for i in range(n_filter):
            left = int(freq_points[i])
            center = int(freq_points[i + 1])
            right = int(freq_points[i + 2])
            
            # Rising edge
            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            # Falling edge
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)
        
        return filters
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spec),
            n_mfcc=self.n_mfcc
        )
        
        if self.normalize:
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        return mfcc
    
    def extract_lfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract LFCC features."""
        # Compute STFT
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        
        # Get power spectrum
        power_spec = np.abs(D) ** 2
        
        # Apply linear filterbank
        filtered_spec = np.dot(self.linear_basis, power_spec)
        
        # Apply log and DCT
        log_spec = np.log(filtered_spec + 1e-8)
        lfcc = scipy.fftpack.dct(log_spec, axis=0, norm='ortho')
        
        # Take first n_lfcc coefficients
        lfcc = lfcc[:self.n_lfcc, :]
        
        if self.normalize:
            lfcc = (lfcc - np.mean(lfcc)) / (np.std(lfcc) + 1e-8)
        
        return lfcc
    
    def extract_features(
        self,
        audio: Union[np.ndarray, str, Path],
        return_tensor: bool = True
    ) -> Union[np.ndarray, Tensor]:
        """Extract features from audio."""
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(audio, sr=self.sample_rate)
        
        # Extract features based on type
        if self.feature_type == FeatureType.MFCC:
            features = self.extract_mfcc(audio)
        elif self.feature_type == FeatureType.LFCC:
            features = self.extract_lfcc(audio)
        else:  # MEL spectrogram
            features = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            features = librosa.power_to_db(features)
            
            if self.normalize:
                features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Convert to tensor if requested
        if return_tensor:
            features = torch.from_numpy(features).float()
            features = features.unsqueeze(0)  # Add channel dimension
            
        return features

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Initialize extractors
    mfcc_extractor = FeatureExtractor(feature_type="mfcc")
    lfcc_extractor = FeatureExtractor(feature_type="lfcc")
    mel_extractor = FeatureExtractor(feature_type="mel")
    
    # Process example file
    audio_path = "e:/PWr/deepfakes/datasets/track1_2-train/Track1.2/train/wav/ADD2023_T1.2_T_00000000.wav"
    
    try:
        # Extract all features without tensor conversion
        mfcc_features = mfcc_extractor.extract_features(audio_path, return_tensor=False)
        lfcc_features = lfcc_extractor.extract_features(audio_path, return_tensor=False)
        mel_features = mel_extractor.extract_features(audio_path, return_tensor=False)
        
        # Remove channel dimension for plotting if present
        if mfcc_features.ndim == 3:
            mfcc_features = mfcc_features.squeeze(0)
        if lfcc_features.ndim == 3:
            lfcc_features = lfcc_features.squeeze(0)
        if mel_features.ndim == 3:
            mel_features = mel_features.squeeze(0)
        
        # Plot features
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot Mel Spectrogram
        img1 = librosa.display.specshow(
            mel_features,
            x_axis='time',
            y_axis='mel',
            sr=mel_extractor.sample_rate,
            hop_length=mel_extractor.hop_length,
            fmin=mel_extractor.fmin,
            fmax=mel_extractor.fmax,
            ax=ax1
        )
        ax1.set_title('Mel Spectrogram')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # Plot MFCC
        img2 = librosa.display.specshow(
            mfcc_features,
            x_axis='time',
            sr=mfcc_extractor.sample_rate,
            hop_length=mfcc_extractor.hop_length,
            ax=ax2
        )
        ax2.set_title('MFCC')
        fig.colorbar(img2, ax=ax2, format='%+2.0f')
        
        # Plot LFCC
        img3 = librosa.display.specshow(
            lfcc_features,
            x_axis='time',
            sr=lfcc_extractor.sample_rate,
            hop_length=lfcc_extractor.hop_length,
            ax=ax3
        )
        ax3.set_title('LFCC')
        fig.colorbar(img3, ax=ax3, format='%+2.0f')
        
        plt.tight_layout()
        plt.show()
        
        # Print shapes before channel dimension addition
        print(f"Mel Spectrogram shape: {mel_features.shape}")
        print(f"MFCC shape: {mfcc_features.shape}")
        print(f"LFCC shape: {lfcc_features.shape}")
        
    except FileNotFoundError:
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
