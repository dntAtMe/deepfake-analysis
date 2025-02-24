"""Utility script to generate and save spectrograms from WAV files."""
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import torch

from audio_processing import AudioToSpectrogram

def load_labels(label_file: Path) -> Dict[str, str]:
    """Load labels from file into a dictionary."""
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split()
            labels[filename] = label
    return labels

def process_dataset(
    wav_dir: Path,
    label_file: Path,
    output_dir: Path,
    converter: AudioToSpectrogram
) -> Tuple[List[Path], List[int]]:
    """Process all WAV files and save spectrograms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(label_file)
    
    processed_files = []
    label_indices = []
    
    # Get total file count for progress tracking
    wav_files = sorted(wav_dir.glob("*.wav"))
    total_files = len(wav_files)
    
    print(f"\nProcessing {total_files} WAV files...")
    
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for wav_path in wav_files:
            if wav_path.name not in labels:
                print(f"\nWarning: No label found for {wav_path.name}")
                continue
                
            try:
                spec = converter(wav_path, return_tensor=True)
                output_path = output_dir / f"{wav_path.stem}.pt"
                torch.save(spec, output_path)
                
                # Store absolute path
                processed_files.append(str(output_path.absolute()))
                label_indices.append(1 if labels[wav_path.name] == "genuine" else 0)
                
                pbar.update(1)
                pbar.set_postfix({"Current": wav_path.name})
                
            except Exception as e:
                print(f"\nError processing {wav_path.name}: {e}")
    
    return processed_files, label_indices

def main():
    try:
        # Try absolute path first
        base_dir = Path("/e:/PWr/deepfakes")
        if not base_dir.exists():
            # Fallback to relative path
            base_dir = Path.cwd()
    except Exception:
        base_dir = Path.cwd()
        
    # Setup paths using resolved paths
    dataset_dir = base_dir / "datasets/track1_2-train/Track1.2/train"
    wav_dir = dataset_dir / "wav"
    label_file = dataset_dir / "label.txt"
    output_dir = dataset_dir / "spect"
    
    # Convert to absolute and resolve any .. or . in paths
    wav_dir = wav_dir.resolve()
    label_file = label_file.resolve()
    output_dir = output_dir.resolve()
    
    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify paths exist
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}\nCurrent directory: {Path.cwd()}")
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
        
    print("Working with paths:")
    print(f"Base directory: {base_dir}")
    print(f"WAV directory: {wav_dir}")
    print(f"Label file: {label_file}")
    print(f"Output directory: {output_dir}")
    
    # Initialize converter
    converter = AudioToSpectrogram(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        target_length=404
    )
    
    # Process files
    processed_files, labels = process_dataset(wav_dir, label_file, output_dir, converter)
    
    # Create and save metadata
    metadata = {
        'files': processed_files,
        'labels': labels,
        'label_mapping': {0: 'fake', 1: 'genuine'},
        'params': {
            'sample_rate': converter.sample_rate,
            'n_fft': converter.n_fft,
            'hop_length': converter.hop_length,
            'n_mels': converter.n_mels,
            'target_length': converter.target_length
        }
    }
    
    metadata_path = output_dir / 'metadata.pt'
    torch.save(metadata, metadata_path)
    print(f"\nSaved metadata to: {metadata_path}")
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total files processed: {len(processed_files)}")
    print(f"Genuine samples: {sum(labels)}")
    print(f"Fake samples: {len(labels) - sum(labels)}")
    
    # Save a sample visualization
    try:
        import matplotlib.pyplot as plt
        sample_spec = torch.load(processed_files[0])
        plt.figure(figsize=(10, 4))
        plt.imshow(sample_spec[0], aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Sample Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel bands')
        plt.savefig(output_dir / 'sample_spectrogram.png')
        plt.close()
    except Exception as e:
        print(f"Could not save visualization: {e}")

if __name__ == "__main__":
    main()
