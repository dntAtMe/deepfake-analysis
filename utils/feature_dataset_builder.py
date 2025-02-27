"""
Script for extracting audio features and creating feature datasets.
"""
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Fix imports to work when run as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.feature_extractor import FeatureExtractor


class FeatureDatasetBuilder:
    """Build datasets of audio features from directories of audio files."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_lfcc: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        feature_types: List[str] = ["mfcc", "lfcc"],
        normalize: bool = True,
        device: str = "cpu",
    ):
        """Initialize dataset builder.
        
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_lfcc: Number of LFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            feature_types: List of feature types to extract
            normalize: Whether to normalize features
            device: Computation device
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_types = feature_types
        self.normalize = normalize
        self.device = device
        
        # Initialize feature extractors for each type
        self.extractors = {}
        for feat_type in feature_types:
            self.extractors[feat_type] = FeatureExtractor(
                feature_type=feat_type,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mfcc=n_mfcc,
                n_lfcc=n_lfcc,
                normalize=normalize,
                device=device
            )
    
    def _get_file_label(self, filepath: str, label_dict: Optional[Dict[str, int]] = None) -> Union[int, str]:
        """Extract label from filepath or lookup in label_dict.
        
        Args:
            filepath: Path to audio file
            label_dict: Dictionary mapping filename patterns to labels
            
        Returns:
            Label as string or int
        """
        if label_dict is not None:
            filename = os.path.basename(filepath)
            for pattern, label in label_dict.items():
                if pattern in filename:
                    return label
        
        # Default: use parent directory name as label
        return Path(filepath).parent.name
    
    def _extract_features_from_file(
        self, 
        filepath: str
    ) -> Dict[str, np.ndarray]:
        """Extract all feature types from a single audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        for feat_type in self.feature_types:
            feat = self.extractors[feat_type].extract_features(
                filepath, return_tensor=False
            )
            
            # Average across time to get fixed-length representation
            feat_avg = np.mean(feat, axis=1)
            
            # Store features
            features[feat_type] = feat_avg
            
        return features
    
    def build_dataset(
        self,
        audio_dir: str,
        file_pattern: str = "*.wav",
        label_dict: Optional[Dict[str, int]] = None,
        max_files: Optional[int] = None,
        include_paths: bool = True
    ) -> pd.DataFrame:
        """Build dataset from directory of audio files.
        
        Args:
            audio_dir: Directory containing audio files
            file_pattern: Glob pattern for finding audio files
            label_dict: Dictionary mapping filename patterns to labels
            max_files: Maximum number of files to process
            include_paths: Whether to include file paths in DataFrame
            
        Returns:
            DataFrame containing extracted features
        """
        # Find audio files
        audio_paths = glob.glob(os.path.join(audio_dir, file_pattern))
        if max_files:
            audio_paths = audio_paths[:max_files]
        
        data = []
        
        # Process files with tqdm progress bar
        for filepath in tqdm(audio_paths, desc="Extracting features"):
            try:
                # Get label
                label = self._get_file_label(filepath, label_dict)
                
                # Extract features
                features = self._extract_features_from_file(filepath)
                
                # Create record
                record = {"label": label}
                
                # Add filepath if requested
                if include_paths:
                    record["filepath"] = filepath
                
                # Add flattened feature values with prefixes
                for feat_type, feat_values in features.items():
                    for i, val in enumerate(feat_values):
                        record[f"{feat_type}_{i}"] = val
                
                data.append(record)
                
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str, format: str = "csv"):
        """Save dataset to disk.
        
        Args:
            df: DataFrame to save
            output_path: Path to save file
            format: Format to save (csv or pickle)
        """
        if format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "pickle":
            df.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Dataset saved to {output_path}")
        

if __name__ == "__main__":
    from env_setup import setup_environment
    setup_environment()

    # Example usage
    data_dir = "e:/PWr/deepfakes/datasets/track1_2-train/Track1.2/train/wav"
    output_dir = "e:/PWr/deepfakes/features"
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Define label mapping based on filename patterns
    label_dict = {
        "bonafide": 0,  # Bonafide samples
        "spoof": 1       # Spoofed samples
    }
    
    # Create feature dataset builder
    feature_builder = FeatureDatasetBuilder(
        feature_types=["mfcc", "lfcc"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Build dataset (limit to 10 files for testing)
    print("Building dataset...")
    df = feature_builder.build_dataset(
        audio_dir=data_dir,
        file_pattern="*.wav",
        label_dict=label_dict,
        max_files=10,  # Remove or set to None for full dataset
        include_paths=True
    )
    
    # Display dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"First few rows:")
    print(df.head())
    
    # Save dataset
    feature_builder.save_dataset(
        df=df,
        output_path=os.path.join(output_dir, "audio_features.csv"),
        format="csv"
    )
    
    # Also save as pickle for faster loading
    feature_builder.save_dataset(
        df=df,
        output_path=os.path.join(output_dir, "audio_features.pkl"),
        format="pickle"
    )
    
    print("Feature extraction completed!")
