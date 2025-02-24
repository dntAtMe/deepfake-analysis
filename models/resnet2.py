"""
RawNet: Deep Neural Network for audio deepfake detection.
Original implementation by Hemlata Tak (tak@eurecom.fr)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResNet2Config:
    """Configuration for ResNet2/RawNet model."""
    filts: List[List[int]]
    first_conv: int
    in_channels: int
    gru_node: int
    nb_gru_layer: int
    nb_fc_node: int
    nb_classes: int
    
    def validate(self) -> None:
        """Validates the configuration parameters."""
        assert len(self.filts) >= 3, "Need at least 3 filter configurations"
        assert self.first_conv > 0, "First conv size must be positive"
        assert self.in_channels > 0, "Input channels must be positive"
        assert self.gru_node > 0, "GRU nodes must be positive"
        assert self.nb_gru_layer > 0, "GRU layers must be positive"
        assert self.nb_fc_node > 0, "FC nodes must be positive"
        assert self.nb_classes > 0, "Need at least 1 class"


class SincConv(nn.Module):
    """Sinc-based convolution for raw audio processing."""
    
    def __init__(self, device: str, out_channels: int, kernel_size: int,
                 in_channels: int = 1, sample_rate: int = 16000) -> None:
        super().__init__()
        assert in_channels == 1, "SincConv only supports one input channel"
        assert kernel_size % 2 != 0, "Kernel size must be odd"
        
        self.device = device
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Mel initialization of filterbanks
        NFFT = 512
        f = int(sample_rate/2) * np.linspace(0, 1, int(NFFT/2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(fmel.min(), fmel.max(), out_channels + 1)
        self.mel = self.to_hz(filbandwidthsmel)
        self.hsupp = torch.arange(-(kernel_size-1)/2, (kernel_size-1)/2 + 1)
        self.band_pass = torch.zeros(out_channels, kernel_size)
        
    @staticmethod
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.mel)-1):
            fmin, fmax = self.mel[i], self.mel[i+1]
            hHigh = (2*fmax/self.sample_rate) * np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i,:] = torch.Tensor(np.hamming(self.kernel_size)) * torch.Tensor(hideal)
        
        band_pass_filter = self.band_pass.to(self.device)
        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters)


class Residual_block(nn.Module):
    """1D Residual block with batch normalization and max pooling."""
    
    def __init__(self, nb_filts: List[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        
        self.bn1 = None if first else nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1)
        
        # Optional downsample path
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], 1)
        
        self.mp = nn.MaxPool1d(3)
    
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


class ResNet2(nn.Module):
    """ResNet2/RawNet architecture for audio deepfake detection."""
    
    def __init__(self, config: ResNet2Config, device: str = "cuda", input_type: str = "raw") -> None:
        super().__init__()
        config.validate()
        
        self.device = device
        self.input_type = input_type
        
        # Adjust first layer based on input type
        if input_type == "spectrogram":
            self.first_layer = nn.Conv2d(1, config.filts[0], kernel_size=3, padding=1)
        else:
            self.first_layer = SincConv(
                device=device,
                out_channels=config.filts[0],
                kernel_size=config.first_conv,
                in_channels=config.in_channels
            )
        
        # Rest of the architecture
        self.first_bn = nn.BatchNorm1d(config.filts[0])
        self.selu = nn.SELU(inplace=True)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            Residual_block(config.filts[1], first=True),
            Residual_block(config.filts[1]),
            Residual_block(config.filts[2]),
            Residual_block([config.filts[2][1], config.filts[2][1]]),
            Residual_block([config.filts[2][1], config.filts[2][1]]),
            Residual_block([config.filts[2][1], config.filts[2][1]])
        ])
        
        # Attention mechanism
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.attention = nn.ModuleList([
            self._make_attention_fc(f[-1]) for f in 
            [config.filts[1], config.filts[1], config.filts[2]] + 
            [[config.filts[2][1], config.filts[2][1]]]*3
        ])
        
        # GRU and final layers
        self.bn_before_gru = nn.BatchNorm1d(config.filts[2][-1])
        self.gru = nn.GRU(
            config.filts[2][-1],
            config.gru_node,
            config.nb_gru_layer,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.gru_node, config.nb_fc_node),
            nn.Linear(config.nb_fc_node, config.nb_classes)
        )
        
    def _make_attention_fc(self, features: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(features, features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input types
        if self.input_type == "spectrogram":
            # Input is already in spectrogram format (B, 1, F, T)
            x = self.first_layer(x)
            x = x.transpose(2, 3)  # Convert to (B, C, T, F) for 1D operations
            x = x.flatten(2)  # Combine frequency bins
        else:
            # Original raw audio processing
            x = x.view(x.shape[0], 1, -1)
            x = self.first_layer(x)
            x = F.max_pool1d(torch.abs(x), 3)
        
        x = self.selu(self.first_bn(x))
        
        # Residual blocks with attention
        for i, (block, attention) in enumerate(zip(self.blocks, self.attention)):
            x_block = block(x)
            y = self.avgpool(x_block).flatten(1)
            y = torch.sigmoid(attention(y)).view(y.size(0), y.size(1), -1)
            x = x_block * y + y
        
        # GRU processing
        x = self.selu(self.bn_before_gru(x))
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        
        # Classification
        x = self.classifier(x[:, -1, :])
        return F.log_softmax(x, dim=1)


def get_default_config(input_type: str = "raw") -> ResNet2Config:
    """Returns default configuration for ResNet2."""
    if input_type == "spectrogram":
        return ResNet2Config(
            filts=[32, [32, 32], [32, 128], [128, 128]],  # Adjusted for spectrograms
            first_conv=3,  # Not used for spectrograms
            in_channels=1,
            gru_node=1024,
            nb_gru_layer=3,
            nb_fc_node=1024,
            nb_classes=2
        )
    else:
        return ResNet2Config(
            filts=[20, [20, 20], [20, 128], [128, 128]],
            first_conv=251,
            in_channels=1,
            gru_node=1024,
            nb_gru_layer=3,
            nb_fc_node=1024,
            nb_classes=2
        )


if __name__ == "__main__":
    import time
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
    
    model = ResNet2(config, device=device).to(device)
    
    # Raw audio input shape
    input_shape = (1, 64000)  # ~4 seconds of audio at 16kHz
    stats = benchmark_model(
        model=model,
        input_shape=input_shape,
        samples_count=100,
        device=device
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Benchmark results (seconds):")
    for key, value in stats.items():
        print(f"{key:>6}: {value:.6f}")
