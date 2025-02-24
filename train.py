"""Training script for SpecRNet model."""
import argparse
from pathlib import Path
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.resnet2 import ResNet2, get_default_config as get_resnet_config
from models.specrnet import SpecRNet, get_default_config as get_specrnet_config
from models.baseline_cnn import BaselineCNN, get_default_config as get_cnn_config
from utils.dataset import SpectrogramDataset
from utils.metrics import get_metrics
from utils.raw_audio_dataset import RawAudioDataset

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float, dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with tqdm(loader, desc="Training") as pbar:
        for specs, labels in pbar:
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Store predictions and labels
            probs = torch.softmax(outputs, dim=1)
            all_predictions.append(probs.detach())
            all_labels.append(labels)
            
            total_loss += loss.item()
            
            # Calculate running metrics
            predictions = torch.cat(all_predictions)
            true_labels = torch.cat(all_labels)
            metrics = get_metrics(predictions, true_labels)
            
            # Update progress bar with all metrics
            _, predicted = predictions.max(1)
            accuracy = predicted.eq(true_labels).sum().item() / len(true_labels)
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(predictions):.3f}',
                'acc': f'{accuracy*100:.1f}%',
                'eer': f'{metrics["eer"]:.1f}%',
                'auc': f'{metrics["auc"]:.3f}'
            })
    
    return total_loss / len(loader), accuracy, metrics

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, dict]:
    """Validate the model and compute metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for specs, labels in pbar:
                specs, labels = specs.to(device), labels.to(device)
                
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                probs = torch.softmax(outputs, dim=1)
                all_predictions.append(probs)
                all_labels.append(labels)
                
                total_loss += loss.item()
                
                # Calculate running metrics
                predictions = torch.cat(all_predictions)
                true_labels = torch.cat(all_labels)
                metrics = get_metrics(predictions, true_labels)
                
                # Update progress bar with all metrics
                _, predicted = predictions.max(1)
                accuracy = predicted.eq(true_labels).sum().item() / len(true_labels)
                
                pbar.set_postfix({
                    'loss': f'{total_loss/len(predictions):.3f}',
                    'acc': f'{accuracy*100:.1f}%',
                    'eer': f'{metrics["eer"]:.1f}%',
                    'auc': f'{metrics["auc"]:.3f}'
                })
    
    return total_loss / len(loader), accuracy, metrics

def get_model(model_name: str, device: str, input_type: str = "spectrogram"):
    """Get model and its configuration based on name."""
    if model_name == "specrnet":
        config = get_specrnet_config()
        model = SpecRNet(config, device=device)
        num_classes = config.nb_classes
    elif model_name == "resnet":
        config = get_resnet_config(input_type=input_type)
        model = ResNet2(config, device=device, input_type=input_type)
        num_classes = config.nb_classes
    elif model_name == "cnn":
        config = get_cnn_config()
        model = BaselineCNN(config)
        num_classes = config.num_classes  # Different attribute name
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, config, num_classes

def get_dataset(metadata_path: Path, input_type: str = "spectrogram"):
    """Get appropriate dataset based on input type."""
    try:
        if (input_type == "raw"):
            print("Loading raw audio dataset...")
            dataset = RawAudioDataset(
                metadata_path=metadata_path,
                sample_rate=16000,
                duration=4.0
            )
            print(f"Found {len(dataset)} audio files")
            # Verify first file can be loaded
            _, _ = dataset[0]
            return dataset
        else:
            return SpectrogramDataset(metadata_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", type=str, 
                       choices=["specrnet", "resnet", "cnn"], 
                       default="specrnet", 
                       help="Model architecture to use")
    parser.add_argument("--input-type", type=str, choices=["spectrogram", "raw"],
                       default="spectrogram", help="Input type to use")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    # Verify compatible model and input type
    if args.model == "specrnet" and args.input_type == "raw":
        raise ValueError("SpecRNet only supports spectrogram input")
    
    # Configuration
    device = "cpu"  # Force CPU usage
    print(f"\nUsing device: {device}")
    print(f"Selected model: {args.model}")
    
    # Setup paths using absolute paths
    try:
        base_dir = Path("/e:/PWr/deepfakes")
        if not base_dir.exists():
            base_dir = Path.cwd()
    except Exception:
        base_dir = Path.cwd()
        
    data_dir = base_dir / "datasets/track1_2-train/Track1.2/train"
    metadata_path = data_dir / "spect/metadata.pt"
    save_dir = base_dir / "checkpoints"
    
    # Verify paths exist
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please run generate_spectrograms.py first to preprocess the audio files."
        )
    
    save_dir.mkdir(exist_ok=True)
    
    # Load appropriate dataset with error handling
    print(f"Loading dataset (type: {args.input_type})...")
    try:
        dataset = get_dataset(metadata_path, args.input_type)
        label_counts = dataset.get_label_counts()
        print("Label distribution:", label_counts)
    except Exception as e:
        print(f"Error: Failed to load dataset - {str(e)}")
        return
    
    # Verify number of classes
    num_classes = len(label_counts)
    print(f"Number of classes: {num_classes}")
    
    # Reduce batch size for CPU training
    batch_size = min(args.batch_size, 16)  # Smaller batches for CPU
    print(f"Using batch size: {batch_size}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Reduced for CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Reduced for CPU
    )
    
    # Get model
    model, config, num_classes = get_model(args.model, device, input_type=args.input_type)
    model = model.to(device)
    
    print("\nModel configuration:")
    print(f"Architecture: {args.model}")
    print(f"Input type: {args.input_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    
    # Training loop
    best_metrics = {
        'val_acc': 0,
        'eer': float('inf'),
        'auc': 0
    }
    
    print(f"\nStarting training on {device}...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate with metrics
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model based on multiple metrics
        improved = (
            val_acc > best_metrics['val_acc'] or
            val_metrics['eer'] < best_metrics['eer'] or
            val_metrics['auc'] > best_metrics['auc']
        )
        
        if improved:
            best_metrics.update({
                'val_acc': val_acc,
                'eer': val_metrics['eer'],
                'auc': val_metrics['auc']
            })
            
            # Create checkpoint name base
            checkpoint_base = f"{args.model}_epoch_{epoch+1:03d}_acc_{val_acc*100:.2f}_eer_{val_metrics['eer']:.2f}"
            
            # Save checkpoint with detailed information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': {
                    'best': best_metrics,
                    'current': {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'eer': val_metrics['eer'],
                        'auc': val_metrics['auc']
                    }
                },
                'config': config,
                'model_name': args.model,
                'input_type': args.input_type,
                'training_args': vars(args),
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }
            
            # Save weights
            torch.save(checkpoint, save_dir / f"{checkpoint_base}.pt")
            
            # For SpecRNet, also save architecture
            if args.model == "specrnet":
                architecture_path = save_dir / f"{checkpoint_base}_arch.json"
                model.save_architecture(str(architecture_path))
            
            # Save as latest best model
            torch.save(checkpoint, save_dir / f"{args.model}_best.pt")
            if args.model == "specrnet":
                model.save_architecture(str(save_dir / f"{args.model}_best_arch.json"))
            
            print(f"New best model saved as {checkpoint_base}!")
        
        # Print results
        print(f"\nResults:")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        print(f"EER: {val_metrics['eer']:.2f}%, AUC: {val_metrics['auc']:.4f}")
        print(f"Best - Acc: {best_metrics['val_acc']*100:.2f}%, "
              f"EER: {best_metrics['eer']:.2f}%, "
              f"AUC: {best_metrics['auc']:.4f}")

if __name__ == "__main__":
    main()
