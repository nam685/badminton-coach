"""TrackNetV3 model definition, vendored (near-verbatim) from RacketVision — see LICENSE_NOTICE.md.

Architecture and forward-pass logic are unchanged from upstream `source/BallTrack/model/tracknet_v3.py`
+ `model/loss_utils.py`; only the training-loss branch is trimmed (this codebase only ever runs
inference against the pretrained `balltrack_best.pth` checkpoint) and constants from upstream's
`configs/tracknetv3_base.py` are inlined here instead of vendoring mmengine's Config machinery for a
handful of scalars.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS

# From RacketVision's source/BallTrack/configs/tracknetv3_base.py
SEQ_LEN = 4
TRACKNET_WIDTH = 512
TRACKNET_HEIGHT = 288
TRACKNET_IN_DIM = 3 * (SEQ_LEN + 1)  # median frame + seq_len frames, 3 channels each
TRACKNET_OUT_DIM = SEQ_LEN
TRACKNET_LAST_ONLY = True


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Triple2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


@MODELS.register_module(name="TrackNetV3")
class TrackNetV3(BaseModel):
    def __init__(
        self,
        in_dim: int = TRACKNET_IN_DIM,
        out_dim: int = TRACKNET_OUT_DIM,
        mixup: bool = True,
        alpha: float = 0.5,
        last_only: bool = TRACKNET_LAST_ONLY,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.mixup = mixup
        self.last_only = last_only
        self.down_block_1 = Double2DConv(in_dim, d_model)
        self.down_block_2 = Double2DConv(d_model, d_model * 2)
        self.down_block_3 = Triple2DConv(d_model * 2, d_model * 4)
        self.bottleneck = Triple2DConv(d_model * 4, d_model * 8)
        self.up_block_1 = Triple2DConv(d_model * (8 + 4), d_model * 4)
        self.up_block_2 = Double2DConv(d_model * (4 + 2), d_model * 2)
        self.up_block_3 = Double2DConv(d_model * (2 + 1), d_model)
        self.predictor = nn.Conv2d(d_model, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block_1(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
        x2 = self.down_block_2(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
        x3 = self.down_block_3(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)
        x = self.bottleneck(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Inference-only forward: returns the (N, 1, H, W) heatmap for the last frame in each
        sequence (upstream's `last_only=True` branch; training/loss modes are not implemented here)."""
        x = self._forward(frames)
        if self.last_only:
            x = x[:, -1:, :, :]
        return x
