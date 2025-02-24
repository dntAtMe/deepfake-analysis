"""Main script for training and evaluating SpecRNet on deepfake detection."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

from models.specrnet import SpecRNet, get_default_config
from utils.dataset import SpectrogramDataset

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    save_dir: Path = Path("checkpoints")
) -> None:
    """Run training and validation loops."""
    save_dir.mkdir(exist_ok=True)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for specs, labels in pbar:
                specs, labels = specs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{train_loss/train_total:.3f}',
                    'acc': f'{100.*train_correct/train_total:.1f}%'
                })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for specs, labels in pbar:
                    specs, labels = specs.to(device), labels.to(device)
                    
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{val_loss/val_total:.3f}',
                        'acc': f'{100.*val_correct/val_total:.1f}%'
                    })
        
        # Calculate epoch statistics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_dir / 'best_model.pt')
            print(f"New best model saved! Val Acc: {val_acc*100:.2f}%")

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Setup paths
    data_dir = Path("datasets/track1_2-train/Track1.2/train")
    metadata_path = data_dir / "spect/metadata.pt"
    save_dir = Path("checkpoints")
    
    # Load dataset
    print("Loading dataset...")
    dataset = SpectrogramDataset(metadata_path)
    print("Class distribution:", dataset.get_label_counts())
    
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
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    config = get_default_config()
    model = SpecRNet(config, device=device).to(device)
    
    # Train model
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir=save_dir
    )

if __name__ == "__main__":
    main()
