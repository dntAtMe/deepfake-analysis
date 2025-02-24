"""Utilities for audio processing and spectrogram generation."""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import librosa
import librosa.display
import scipy.signal


class AudioToSpectrogram:
    """Converts audio files to spectrograms using librosa."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,        # Increased for better frequency resolution
        hop_length: int = 256,     # Adjusted for 16ms hop at 16kHz
        n_mels: int = 80,
        target_length: int = 404,
        fmin: int = 20,
        fmax: int = 8000,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        bits_per_sample: int = 16  # For 256 kbps
    ):
        """Initialize the converter with specific parameters."""
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_length = target_length
        self.fmin = fmin
        self.fmax = fmax
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.bits_per_sample = bits_per_sample
        
        # Calculate optimal window size based on bit rate
        self.window_length = self.n_fft
        self.bytes_per_frame = bits_per_sample // 8
        
        # Pre-compute mel filterbank with higher frequency resolution
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=True  # Use HTK formula for better precision
        )

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """Load and preprocess audio file, returning audio and its duration."""
        # Get duration before loading
        duration = librosa.get_duration(path=str(audio_path))
        
        # Load audio with specific sample width
        audio, _ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            res_type='kaiser_best'  # Higher quality resampling
        )
        
        # Handle audio duration
        if duration < self.min_duration:
            repeats = int(np.ceil(self.min_duration / duration))
            audio = np.tile(audio, repeats)
            duration = len(audio) / self.sample_rate
        elif duration > self.max_duration:
            audio = audio[:int(self.max_duration * self.sample_rate)]
            duration = self.max_duration
        
        # Apply pre-emphasis with higher coefficient for high-quality audio
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Normalize with bit depth consideration
        max_val = float(2 ** (self.bits_per_sample - 1))
        audio = np.clip(audio, -1.0, 1.0) * max_val
        audio = audio / max_val
        
        return audio, duration

    def generate_spectrogram(
        self,
        audio: np.ndarray,
        duration: float,
        normalize: bool = False  # Changed default to False
    ) -> np.ndarray:
        """Convert audio to mel spectrogram with natural dB scale."""
        # Generate STFT with higher precision settings
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window='hann',
            center=True,
            pad_mode='reflect',
            dtype=np.float32
        )
        
        # Convert to power spectrogram
        S = np.abs(D) ** 2.0
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, S)
        
        # Convert to log scale preserving natural dB range
        mel_spec = librosa.power_to_db(
            mel_spec,
            ref=np.max,
            amin=1e-10,
            top_db=80.0  # Standard dynamic range
        )
        
        # Remove normalization step to preserve dB scale
        return mel_spec
    
    def pad_or_truncate(self, spec: np.ndarray) -> np.ndarray:
        """Adjust spectrogram to target length."""
        curr_length = spec.shape[1]
        
        if curr_length > self.target_length:
            # Truncate to target length
            start = (curr_length - self.target_length) // 2
            spec = spec[:, start:start + self.target_length]
        elif curr_length < self.target_length:
            # Pad to target length
            pad_width = self.target_length - curr_length
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            spec = np.pad(spec, ((0, 0), (left_pad, right_pad)))
            
        return spec
    
    def visualize_spectrogram(
        self,
        spec: np.ndarray,
        duration: float,
        output_path: Optional[Path] = None
    ) -> None:
        """Visualize spectrogram with natural dB scale."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        librosa.display.specshow(
            spec[0],
            y_axis='mel',
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Set proper time axis based on actual duration
        plt.xlim(0, duration)
        tick_step = 1.0 if duration < 10 else 2.0
        plt.xticks(np.arange(0, duration + tick_step, tick_step))
        
        plt.colorbar(format='%+2.0f dB')  # Will now show actual dB values
        plt.title(f'Mel Spectrogram ({duration:.1f}s)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Mel Frequency (Hz)')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def __call__(
        self,
        audio_path: Union[str, Path],
        return_tensor: bool = True,
        visualize: bool = False
    ) -> Union[np.ndarray, Tensor]:
        """Process audio file to spectrogram."""
        # Load and process audio
        audio, duration = self.load_audio(audio_path)
        spec = self.generate_spectrogram(audio, duration)
        spec = self.pad_or_truncate(spec)
        
        # Add channel dimension
        spec = np.expand_dims(spec, 0)
        
        if visualize:
            self.visualize_spectrogram(spec, duration)
        
        if return_tensor:
            return torch.from_numpy(spec).float()
        return spec


if __name__ == "__main__":
    # Example usage with visualization
    converter = AudioToSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,  # 10ms hop
    )
    
    try:
        # Replace with actual path
        audio_path = "e:/PWr/deepfakes/datasets/track1_2-train/Track1.2/train/wav/ADD2023_T1.2_T_00000000.wav"
        spec = converter(
            audio_path,
            return_tensor=False,
            visualize=True
        )
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Time resolution: {converter.hop_length/converter.sample_rate*1000:.1f}ms")
        print(f"Expected width: {converter.expected_frames} frames")
        print(f"Actual width: {spec.shape[2]} frames")
        
    except FileNotFoundError:
        print("Please provide a valid audio file path")
