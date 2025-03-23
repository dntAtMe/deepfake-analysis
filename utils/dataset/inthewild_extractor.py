import os
import pandas as pd
from typing import Dict, List

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/release_in_the_wild"))
metadata_path = os.path.join(dataset_path, "meta.csv")

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


