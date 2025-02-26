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
import wandb
from datetime import datetime

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
    device: str,
    epoch: int,
    global_step: int,
    log_wandb: bool = False,
    log_interval: int = 10
) -> Tuple[float, float, dict, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    step = 0
    
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
            step += 1
            global_step += 1
            
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
            
            # Log to wandb every log_interval steps
            if log_wandb and step % log_interval == 0:
                batch_metrics = {
                    f"train/{epoch}/step_loss": loss.item(),
                    f"train/{epoch}/step_accuracy": accuracy,
                    f"train/{epoch}/step_eer": metrics["eer"],
                    f"train/{epoch}/step_auc": metrics["auc"],
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                }
                wandb.log(batch_metrics, step=global_step)
    
    return total_loss / len(loader), accuracy, metrics, global_step

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    global_step: int,
    log_wandb: bool = False,
    log_interval: int = 10
) -> Tuple[float, float, dict]:
    """Validate the model and compute metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    step = 0
    
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
                step += 1
                
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
                
                # Log to wandb every log_interval steps
                if log_wandb and step % log_interval == 0:
                    batch_metrics = {
                        f"val/{epoch}/step_loss": loss.item(),
                        f"val/{epoch}/step_accuracy": accuracy,
                        f"val/{epoch}/step_eer": metrics["eer"],
                        f"val/{epoch}/step_auc": metrics["auc"],
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step + step,
                    }
                    wandb.log(batch_metrics, step=global_step + step)
    
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

def get_dataset(metadata_path: Path, input_type: str = "spectrogram", device: str = "cpu"):
    """Get appropriate dataset based on input type."""
    try:
        if (input_type == "raw"):
            print("Loading raw audio dataset...")
            dataset = RawAudioDataset(
                metadata_path=metadata_path,
                sample_rate=16000,
                duration=4.0,
                device=device
            )
            print(f"Found {len(dataset)} audio files")
            # Verify first file can be loaded
            _, _ = dataset[0]
            return dataset
        else:
            return SpectrogramDataset(metadata_path, device=device)
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
    parser.add_argument("--wandb-project", type=str, default="deepfakes",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="Weights & Biases entity (username or team)")
    parser.add_argument("--disable-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Interval for logging steps to wandb")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use for training")
    args = parser.parse_args()
    
    # Verify compatible model and input type
    if args.model == "specrnet" and args.input_type == "raw":
        raise ValueError("SpecRNet only supports spectrogram input")
    
    # Configuration - Replace existing device setting
    device = args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
    print(f"\nUsing device: {device}")
    print(f"Selected model: {args.model}")
    
    # Initialize wandb if not disabled
    run = None
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "model": args.model,
                "input_type": args.input_type,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "device": device,
                "log_interval": args.log_interval,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        )
    
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
        dataset = get_dataset(metadata_path, args.input_type, device=device)
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
    
    # Log model architecture
    if args.model == "specrnet" and run is not None:
        wandb.config.update({"model_config": config.__dict__})
    
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
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, global_step, log_wandb=(run is not None), 
            log_interval=args.log_interval
        )
        
        # Validate with metrics
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, 
            epoch, global_step, log_wandb=(run is not None),
            log_interval=args.log_interval
        )
        
        # Log epoch summary metrics to wandb
        if run is not None:
            wandb.log({
                "epoch": epoch,
                f"train/{epoch}/loss": train_loss,
                f"train/{epoch}/accuracy": train_acc,
                f"train/{epoch}/eer": train_metrics["eer"],
                f"train/{epoch}/auc": train_metrics["auc"],
                f"val/{epoch}/loss": val_loss,
                f"val/{epoch}/accuracy": val_acc,
                f"val/{epoch}/eer": val_metrics["eer"],
                f"val/{epoch}/auc": val_metrics["auc"],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "global_step": global_step
            }, step=global_step)
            
            # Also log to standard metrics for comparison across runs
            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/eer": train_metrics["eer"],
                "train/auc": train_metrics["auc"],
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/eer": val_metrics["eer"],
                "val/auc": val_metrics["auc"],
            }, step=global_step)
        
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
            
            # Full checkpoint path with extension
            checkpoint_path = save_dir / f"{checkpoint_base}.pt"
            
            # Save weights
            torch.save(checkpoint, checkpoint_path)
            
            # For SpecRNet, also save architecture
            if args.model == "specrnet":
                architecture_path = save_dir / f"{checkpoint_base}_arch.json"
                model.save_architecture(str(architecture_path))
            
            # Save as latest best model
            torch.save(checkpoint, save_dir / f"{args.model}_best.pt")
            if args.model == "specrnet":
                model.save_architecture(str(save_dir / f"{args.model}_best_arch.json"))
            
            print(f"New best model saved as {checkpoint_base}!")
            
            # Log best model to wandb
            if run is not None:
                artifact = wandb.Artifact(
                    name=f"model-{run.id}",
                    type="model",
                    description=f"Best model from run {run.id}"
                )
                # Use the full path with extension
                artifact.add_file(str(checkpoint_path))
                run.log_artifact(artifact)
                
                # Log best metrics
                wandb.run.summary.update({
                    "best_val_acc": val_acc,
                    "best_val_eer": val_metrics['eer'],
                    "best_val_auc": val_metrics['auc'],
                    "best_epoch": epoch
                })
        
        # Print results
        print(f"\nResults:")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        print(f"EER: {val_metrics['eer']:.2f}%, AUC: {val_metrics['auc']:.4f}")
        print(f"Best - Acc: {best_metrics['val_acc']*100:.2f}%, "
              f"EER: {best_metrics['eer']:.2f}%, "
              f"AUC: {best_metrics['auc']:.4f}")
    
    # Finish wandb run
    if run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
