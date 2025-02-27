"""Audio feature extraction utilities."""
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import librosa
import torch
from torch import Tensor
import scipy.fftpack
import argparse

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
        normalize: bool = True,
        device: str = "cpu"
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
            device: Computation device ('cpu' or 'cuda')
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
        self.device = device
        
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
            if self.device != "cpu":
                features = features.to(self.device)
            
        return features

    def save_features_plot(
        self,
        features: dict,
        save_path: Union[str, Path],
        filename: str
    ) -> None:
        """Save extracted features as PNG files.
        
        Args:
            features: Dictionary containing 'mel', 'mfcc', and 'lfcc' features
            save_path: Directory to save the plots
            filename: Base filename without extension
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot Mel Spectrogram
        img1 = librosa.display.specshow(
            features['mel'],
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            ax=ax1
        )
        ax1.set_title('Mel Spectrogram')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # Plot MFCC
        img2 = librosa.display.specshow(
            features['mfcc'],
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=ax2
        )
        ax2.set_title('MFCC')
        fig.colorbar(img2, ax=ax2, format='%+2.0f')
        
        # Plot LFCC
        img3 = librosa.display.specshow(
            features['lfcc'],
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=ax3
        )
        ax3.set_title('LFCC')
        fig.colorbar(img3, ax=ax3, format='%+2.0f')
        
        plt.tight_layout()
        
        # Save plot
        save_file = save_path / f"{filename}_features.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

    def save_individual_features(
        self,
        features: dict,
        save_path: Union[str, Path],
        filename: str
    ) -> None:
        """Save each feature type as a separate image without plot decorations.
        
        Args:
            features: Dictionary containing 'mel', 'mfcc', and 'lfcc' features
            save_path: Directory to save the images
            filename: Base filename without extension
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each feature type separately
        for feature_name, feature_data in features.items():
            # Create figure with only the feature image
            plt.figure(figsize=(12, 4))
            plt.imshow(feature_data, aspect='auto', origin='lower')
            plt.axis('off')  # Remove axes
            
            # Save without plot decorations
            save_file = save_path / f"{filename}_{feature_name}.png"
            plt.savefig(
                save_file,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract and save audio features')
    parser.add_argument('--split', action='store_true', 
                      help='Save features as separate images instead of combined plot')
    args = parser.parse_args()

    # Example usage
    import matplotlib.pyplot as plt
    from env_setup import setup_environment
    setup_environment()
    
    # Use device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize extractors
    mfcc_extractor = FeatureExtractor(feature_type="mfcc", device=device)
    lfcc_extractor = FeatureExtractor(feature_type="lfcc", device=device)
    mel_extractor = FeatureExtractor(feature_type="mel", device=device)
    
    metadata_path = Path("e:/PWr/deepfakes/datasets/track1_2-train/Track1.2/train/spect/metadata.pt")
    data = torch.load(metadata_path)
    file_paths = [Path(p) for p in data['files']]
    labels = data['labels']
    label_mapping = data['label_mapping']

    wav_dir = metadata_path.parent.parent / "wav"
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
    
    # Create output directory for feature plots
    output_dir = Path("e:/PWr/deepfakes/features")
    if args.split:
        output_dir = output_dir / "split"
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # Map spectrogram paths to WAV paths
    wav_paths = [
        wav_dir / f"{Path(p).stem}.wav" 
        for p in data['files']
    ]

    for audio_path in wav_paths:
        try:
            # Extract all features without tensor conversion
            mfcc_features = mfcc_extractor.extract_features(audio_path, return_tensor=False)
            lfcc_features = lfcc_extractor.extract_features(audio_path, return_tensor=False)
            mel_features = mel_extractor.extract_features(audio_path, return_tensor=False)
            
            # Remove channel dimension if present
            if mfcc_features.ndim == 3:
                mfcc_features = mfcc_features.squeeze(0)
            if lfcc_features.ndim == 3:
                lfcc_features = lfcc_features.squeeze(0)
            if mel_features.ndim == 3:
                mel_features = mel_features.squeeze(0)
            
            # Create features dictionary
            features_dict = {
                'mel': mel_features,
                'mfcc': mfcc_features,
                'lfcc': lfcc_features
            }
            
            # Save features based on split argument
            if args.split:
                mel_extractor.save_individual_features(
                    features_dict,
                    output_dir,
                    audio_path.stem
                )
            else:
                mel_extractor.save_features_plot(
                    features_dict,
                    output_dir,
                    audio_path.stem
                )
            
            print(f"Saved features {'separately' if args.split else 'as plot'} for {audio_path.stem}")
            
        except FileNotFoundError:
            print(f"File not found: {audio_path}")
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

