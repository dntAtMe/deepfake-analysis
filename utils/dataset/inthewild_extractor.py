import os
import pandas as pd
import numpy as np
import librosa
import torch
import scipy.fftpack
from typing import Dict, List
from tqdm import tqdm

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/release_in_the_wild"))
metadata_path = os.path.join(dataset_path, "meta.csv")
features_path = os.path.join(dataset_path, "features.csv")

def extract_labels() -> Dict[str, int]:
    """Extract labels from the metadata file.
    
    Returns:
        Dict[str, int]: Dictionary mapping file names to their labels (0 for bona-fide, 1 for spoof)
    """
    df = pd.read_csv(metadata_path)
    # Convert labels to binary: bona-fide -> 0, spoof -> 1
    label_map = {'bona-fide': 0, 'spoof': 1}
    return dict(zip(df['file'], df['label'].map(label_map)))

def extract_audio_files() -> Dict[str, str]:
    """Extract audio files from the metadata file.
    
    Returns:
        Dict[str, str]: Dictionary mapping absolute file paths to file names
    """
    df = pd.read_csv(metadata_path)
    files = df['file'].tolist()
    return {os.path.join(dataset_path, f): f for f in files}

def get_features_dataframe(device: str = "cpu") -> pd.DataFrame:
    """Get the features dataframe from the features file.
    If dataframe is not present, build it and save it to the features file.
    
    Args:
        device: Device to load tensors onto ('cpu' or 'cuda')
        
    Returns:
        pd.DataFrame: DataFrame containing audio files and their labels
    """
    # Check if features file exists
    if os.path.exists(features_path):
        return pd.read_csv(features_path)
    
    # If not, build features and save to file
    df = build_features_dataframe(device)
    df.to_csv(features_path, index=False)
    return df

def build_features_dataframe(device: str = "cpu") -> pd.DataFrame:
    """Build a dataframe from the audio files and their labels.
    Includes MFCCs, LFCCs, MELs, and their labels.

    Args:
        device: Device to load tensors onto ('cpu' or 'cuda')
        
    Returns:
        pd.DataFrame: DataFrame containing audio files and their labels
    """
    # Get labels and audio files
    labels = extract_labels()
    audio_files = extract_audio_files()
    
    # Initialize lists to store features and metadata
    features_list = []
    
    # If no files, return empty DataFrame with correct structure
    if not audio_files:
        return pd.DataFrame(columns=[
            'file', 'label',
            *[f'mfcc_mean_{i}' for i in range(40)],
            *[f'mfcc_std_{i}' for i in range(40)],
            *[f'mel_mean_{i}' for i in range(80)],
            *[f'mel_std_{i}' for i in range(80)],
            *[f'lfcc_mean_{i}' for i in range(40)],
            *[f'lfcc_std_{i}' for i in range(40)]
        ])
    
    # Audio processing parameters
    sr = 16000  # Sample rate
    n_fft = 512  # FFT window size
    hop_length = 160  # Hop length (10ms at 16kHz)
    n_mels = 80  # Number of mel bands
    n_mfcc = 40  # Number of MFCC coefficients
    
    # Process each audio file
    for file_path, file_name in tqdm(audio_files.items(), desc="Extracting features"):
        try:
            # Load audio
            y, _ = librosa.load(file_path, sr=sr)
            
            # Extract features
            # 1. MFCC
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # 2. MEL spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec)
            
            # 3. LFCC (Linear Frequency Cepstral Coefficients)
            # Compute linear spectrogram
            linear_spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            # Apply DCT to get LFCC
            lfcc = scipy.fftpack.dct(np.log(linear_spec + 1e-6), axis=0, norm='ortho')[:n_mfcc]
            
            # Compute statistics for each feature
            feature_dict = {
                'file': file_name,
                'label': labels[file_name],
                
                # MFCC statistics
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                
                # MEL statistics
                'mel_mean': np.mean(mel_spec_db, axis=1),
                'mel_std': np.std(mel_spec_db, axis=1),
                
                # LFCC statistics
                'lfcc_mean': np.mean(lfcc, axis=1),
                'lfcc_std': np.std(lfcc, axis=1)
            }
            
            # Flatten the dictionary
            flat_dict = {}
            for key, value in feature_dict.items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        flat_dict[f"{key}_{i}"] = v
                else:
                    flat_dict[key] = value
            
            features_list.append(flat_dict)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Convert to torch tensors if needed
    if device != "cpu":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        tensor_df = df.copy()
        for col in numeric_cols:
            tensor = torch.tensor(df[col].values, device=device)
            # Store the CPU version in the DataFrame
            df[col] = tensor.cpu().numpy()
    
    return df

# Handle OpenMP runtime duplicate loading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":
    df = get_features_dataframe(device="cuda")

