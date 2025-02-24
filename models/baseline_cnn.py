"""
Baseline CNN model for audio deepfake detection.
Architecture: 3 × {Conv-ReLU-AP-Dropout} + Dense layers
"""
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineCNNConfig:
    """Configuration for Baseline CNN model."""
    input_channels: int
    conv_channels: List[int]
    fc_size: int
    num_classes: int
    dropout_rate: float
    
    def validate(self) -> None:
        """Validates the configuration parameters."""
        assert self.input_channels > 0, "Input channels must be positive"
        assert len(self.conv_channels) == 3, "Need exactly 3 conv layers"
        assert all(c > 0 for c in self.conv_channels), "Channel counts must be positive"
        assert self.fc_size > 0, "FC size must be positive"
        assert self.num_classes > 0, "Need at least 1 class"
        assert 0 <= self.dropout_rate <= 1, "Dropout rate must be between 0 and 1"


class BaselineCNN(nn.Module):
    """Baseline CNN architecture for audio deepfake detection."""
    
    def __init__(self, config: BaselineCNNConfig):
        """Initialize the model."""
        super().__init__()
        config.validate()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.input_channels, config.conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(config.dropout_rate)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.conv_channels[0], config.conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(config.dropout_rate)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(config.conv_channels[1], config.conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(config.dropout_rate)
        )
        
        # Calculate the flattened size after convolutions
        self.flatten_size = self._calculate_flatten_size(
            config.input_channels,
            (80, 404)  # Standard input size (mel_bands, time_steps)
        )
        
        # Dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_size, config.fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate)
        )
        
        # Output layer with softmax
        self.fc2 = nn.Linear(config.fc_size, config.num_classes)
    
    def _calculate_flatten_size(self, channels: int, input_size: tuple) -> int:
        """Calculate the size of flattened features after convolutions."""
        x = torch.zeros(1, channels, *input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel() // x.size(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def get_default_config(input_channels: int = 1) -> BaselineCNNConfig:
    """Returns default configuration for Baseline CNN."""
    return BaselineCNNConfig(
        input_channels=input_channels,
        conv_channels=[32, 64, 128],  # As specified in the architecture
        fc_size=256,                  # As specified in the architecture
        num_classes=2,                # Binary classification
        dropout_rate=0.2              # As specified in the architecture
    )


if __name__ == "__main__":
    # Test model
    config = get_default_config()
    model = BaselineCNN(config)
    
    # Print model summary
    print("Baseline CNN Model Summary:")
    print(f"Configuration:\n{config}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 80, 404)  # Standard input size
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
