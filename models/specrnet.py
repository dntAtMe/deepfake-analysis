"""
SpecRNet: Deep Neural Network for audio deepfake detection.
Source: https://github.com/piotrkawa/specrnet
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn


@dataclass
class SpecRNetConfig:
    """Configuration for SpecRNet model."""
    filts: List[List[int]]
    gru_node: int
    nb_gru_layer: int
    nb_fc_node: int
    nb_classes: int
    input_channels: int = 1

    def validate(self) -> None:
        """Validates the configuration parameters."""
        assert len(self.filts) >= 4, "Need input channels and 3 filter configurations"
        assert isinstance(self.filts[0], int), "First element should be input channels"
        assert all(len(f) == 2 for f in self.filts[1:]), "Each filter config needs 2 values"
        assert self.gru_node > 0, "GRU nodes must be positive"
        assert self.nb_gru_layer > 0, "GRU layers must be positive"
        assert self.nb_fc_node > 0, "FC nodes must be positive"
        assert self.nb_classes > 0, "Need at least 1 class"


class Residual_block2D(nn.Module):
    """2D Residual block with batch normalization and max pooling."""

    def __init__(self, nb_filts: List[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        
        # Only create batch norm if not first layer
        self.bn1 = None if first else nn.BatchNorm2d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], 3, padding=1, stride=1)
        
        # Optional downsample path
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], 1, padding=0, stride=1)
        
        self.mp = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First batch norm and activation
        out = x if self.first else self.lrelu(self.bn1(x))
        
        # Main path
        out = self.conv1(out)
        out = self.lrelu(self.bn2(out))
        out = self.conv2(out)
        
        # Skip connection
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        return self.mp(out + identity)


class SpecRNet(nn.Module):
    """SpecRNet architecture for audio deepfake detection."""

    def __init__(self, config: SpecRNetConfig, device: str = "cuda") -> None:
        super().__init__()
        config.validate()
        
        # Store config for later use
        self.config = config
        self.device = device
        
        # Initial processing
        self.first_bn = nn.BatchNorm2d(config.filts[0])
        self.selu = nn.SELU(inplace=True)
        
        # Residual blocks
        self.block0 = Residual_block2D(config.filts[1], first=True)
        self.block2 = Residual_block2D(config.filts[2])
        self.block4 = Residual_block2D([config.filts[2][1], config.filts[2][1]])
        
        # Attention mechanism
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.ModuleList([
            self._make_attention_fc(f[-1]) 
            for f in [config.filts[1], config.filts[2], config.filts[2]]
        ])
        
        # GRU and final layers
        self.bn_before_gru = nn.BatchNorm2d(config.filts[2][-1])
        self.gru = nn.GRU(
            config.filts[2][-1], 
            config.gru_node,
            config.nb_gru_layer,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.gru_node * 2, config.nb_fc_node * 2),
            nn.Linear(config.nb_fc_node * 2, config.nb_classes)
        )
        
    def _make_attention_fc(self, features: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(features, features))

    def _apply_attention(self, x: torch.Tensor, block_out: torch.Tensor, 
                        attention: nn.Module) -> torch.Tensor:
        y = self.avgpool(block_out).flatten(1)
        y = torch.sigmoid(attention(y)).view(y.size(0), y.size(1), -1, 1)
        return block_out * y + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.selu(self.first_bn(x))
        
        # Residual blocks with attention
        x = self._apply_attention(x, self.block0(x), self.attention[0])
        x = nn.MaxPool2d(2)(x)
        
        x = self._apply_attention(x, self.block2(x), self.attention[1])
        x = nn.MaxPool2d(2)(x)
        
        x = self._apply_attention(x, self.block4(x), self.attention[2])
        x = nn.MaxPool2d(2)(x)
        
        # GRU processing
        x = self.selu(self.bn_before_gru(x))
        x = x.squeeze(-2).permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        
        # Classification
        return self.classifier(x[:, -1, :])
    
    def get_layer_info(self) -> dict:
        """Get serializable information about model layers."""
        return {
            'first_bn': {
                'num_features': self.first_bn.num_features,
                'eps': self.first_bn.eps,
                'momentum': self.first_bn.momentum
            },
            'blocks': {
                'block0': {
                    'input_channels': self.block0.conv1.in_channels,
                    'output_channels': self.block0.conv1.out_channels
                },
                'block2': {
                    'input_channels': self.block2.conv1.in_channels,
                    'output_channels': self.block2.conv1.out_channels
                },
                'block4': {
                    'input_channels': self.block4.conv1.in_channels,
                    'output_channels': self.block4.conv1.out_channels
                }
            },
            'gru': {
                'input_size': self.gru.input_size,
                'hidden_size': self.gru.hidden_size,
                'num_layers': self.gru.num_layers,
                'bidirectional': self.gru.bidirectional
            },
            'classifier': {
                'fc1_in': self.classifier[0].in_features,
                'fc1_out': self.classifier[0].out_features,
                'fc2_in': self.classifier[1].in_features,
                'fc2_out': self.classifier[1].out_features
            }
        }
    
    def save_architecture(self, save_path: str) -> None:
        """Save model architecture to file."""
        architecture = {
            'config': self.config.__dict__,
            'layers': self.get_layer_info()
        }
        with open(save_path, 'w') as f:
            json.dump(architecture, f, indent=4)
    
    @classmethod
    def from_architecture(cls, architecture_path: str, weights_path: str = None):
        """Create model from saved architecture and optionally load weights."""
        with open(architecture_path, 'r') as f:
            architecture = json.load(f)
        
        # Create config from saved dict
        config = SpecRNetConfig(**architecture['config'])
        
        # Create model
        model = cls(config)
        
        # Verify layer dimensions match saved architecture
        current_arch = model.get_layer_info()
        if current_arch != architecture['layers']:
            raise ValueError("Saved architecture doesn't match current model structure")
        
        # Load weights if provided
        if weights_path:
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


def get_default_config(input_channels: int = 1) -> SpecRNetConfig:
    """Returns default configuration for SpecRNet."""
    return SpecRNetConfig(
        filts=[input_channels, [input_channels, 20], [20, 64], [64, 64]],
        gru_node=64,
        nb_gru_layer=2,
        nb_fc_node=64,
        nb_classes=2
    )


if __name__ == "__main__":
    import time
    import numpy as np
    from tqdm import tqdm
    
    def benchmark_model(
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        samples_count: int = 1000,
        device: str = "cuda"
    ) -> Dict[str, float]:
        model.eval()
        durations = []
        
        with torch.no_grad():
            for _ in tqdm(range(samples_count), desc="Running inference"):
                input_tensor = torch.rand(input_shape, device=device)
                start = time.time()
                _ = model(input_tensor)
                end = time.time()
                durations.append(end - start)
                
        durations = np.array(durations)
        return {
            "min": float(durations.min()),
            "max": float(durations.max()),
            "mean": float(durations.mean()),
            "std": float(durations.std())
        }

    # Run benchmark
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_default_config()
    print(f"Configuration:\n{config}")
    print(f"Device: {device}")
    
    model = SpecRNet(config, device=device).to(device)
    
    input_shape = (1, 1, 80, 404)  # Standard input shape
    stats = benchmark_model(
        model=model,
        input_shape=input_shape,
        samples_count=1000,
        device=device
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Benchmark results (seconds):")
    for key, value in stats.items():
        print(f"{key:>6}: {value:.6f}")