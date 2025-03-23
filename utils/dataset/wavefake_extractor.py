import os
from typing import Dict, List

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/track1_2-train/Track1.2/train"))
wav_path = os.path.join(dataset_path, "wav")
labels_path = os.path.join(dataset_path, "label.txt")

def extract_labels() -> Dict[str, int]:
    """
    Extract labels from label.txt file.
    Returns a dictionary mapping wav filenames to their labels (0 for fake, 1 for genuine)
    """
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            filename, label = line.strip().split()
            # Convert label to integer (0 for fake, 1 for genuine)
            labels[filename] = 0 if label.lower() == 'genuine' else 1
    return labels

def extract_audio_files() -> Dict[str, str]:
    """
    Extract wav files from the wav directory.
    Returns a dictionary mapping absolute file paths to filenames.
    Example: {'/path/to/file.wav': 'file.wav'}
    """
    # Get all files in wav directory
    files = os.listdir(wav_path)
    # Filter only .wav files and create dictionary with absolute paths
    wav_files = {
        os.path.join(wav_path, f): f 
        for f in sorted(files) 
        if f.endswith('.wav')
    }
    return wav_files