"""
Functional Core - Model Factory and Definitions

Pure factory functions and class definitions for steganography models.
This module provides the model architecture without I/O side effects.

Model loading (weights from disk) is handled by the imperative shell.
"""

from typing import Protocol, Any, runtime_checkable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import ModelConfig


# ============================================================================
# ILWT53 Wavelet Transform (Helpers)
# ============================================================================

def _reflect_pad_lastdim_1(x: torch.Tensor):
    if x.size(-1) < 2:
        left = x[..., :1]
        right = x[..., -1:]
    else:
        left = x[..., 1:2]
        right = x[..., -2:-1]
    return torch.cat([left, x, right], dim=-1)


def _ilwt53_forward_1d_lastdim(x: torch.Tensor):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    even_ext = _reflect_pad_lastdim_1(even)
    pred = 0.5 * (even_ext[..., :-2] + even_ext[..., 2:])
    d = odd - pred
    d_ext = _reflect_pad_lastdim_1(d)
    upd = 0.25 * (d_ext[..., :-2] + d_ext[..., 2:])
    s = even + upd
    return s, d


def _ilwt53_inverse_1d_lastdim(s: torch.Tensor, d: torch.Tensor):
    d_ext = _reflect_pad_lastdim_1(d)
    upd = 0.25 * (d_ext[..., :-2] + d_ext[..., 2:])
    even = s - upd
    even_ext = _reflect_pad_lastdim_1(even)
    pred = 0.5 * (even_ext[..., :-2] + even_ext[..., 2:])
    odd = d + pred
    last_len = even.size(-1) + odd.size(-1)
    out_shape = list(even.shape)
    out_shape[-1] = last_len
    y = torch.zeros(out_shape, dtype=even.dtype, device=even.device)
    y[..., 0::2] = even
    y[..., 1::2] = odd
    return y


def _apply_forward_1d_along_dim(x: torch.Tensor, dim: int):
    if dim == -1 or dim == x.dim() - 1:
        return _ilwt53_forward_1d_lastdim(x)
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(*perm)
    s, d = _ilwt53_forward_1d_lastdim(x_perm)
    inv_perm = list(range(x_perm.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    s = s.permute(*inv_perm)
    d = d.permute(*inv_perm)
    return s, d


def _apply_inverse_1d_along_dim(s: torch.Tensor, d: torch.Tensor, dim: int):
    if dim == -1 or dim == s.dim() - 1:
        return _ilwt53_inverse_1d_lastdim(s, d)
    perm = list(range(s.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    s_perm = s.permute(*perm)
    d_perm = d.permute(*perm)
    x_perm = _ilwt53_inverse_1d_lastdim(s_perm, d_perm)
    inv_perm = list(range(x_perm.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    x = x_perm.permute(*inv_perm)
    return x


class ILWT53_2D(nn.Module):
    def __init__(self, channels: int):
        super(ILWT53_2D, self).__init__()
        self.channels = channels
        self.padding_info = None

    def _maybe_pad(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        self.padding_info = (pad_h, pad_w)
        return x

    def forward(self, x: torch.Tensor):
        x = self._maybe_pad(x)
        s_row, d_row = _apply_forward_1d_along_dim(x, dim=3)
        LL, LH = _apply_forward_1d_along_dim(s_row, dim=2)
        HL, HH = _apply_forward_1d_along_dim(d_row, dim=2)
        out = torch.cat([LL, LH, HL, HH], dim=1)
        return out

    def inverse(self, z: torch.Tensor):
        b, c4, h2, w2 = z.shape
        c = c4 // 4
        LL, LH, HL, HH = torch.split(z, c, dim=1)
        s_row = _apply_inverse_1d_along_dim(LL, LH, dim=2)
        d_row = _apply_inverse_1d_along_dim(HL, HH, dim=2)
        x = _apply_inverse_1d_along_dim(s_row, d_row, dim=3)
        if self.padding_info is not None:
            pad_h, pad_w = self.padding_info
            if pad_h:
                x = x[:, :, :-pad_h, :]
            if pad_w:
                x = x[:, :, :, :-pad_w]
        return x


# ============================================================================
# Color Space Conversion
# ============================================================================

def rgb_to_ycbcr(x):
    x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    out01 = torch.cat([y, cb, cr], dim=1)
    return out01 * 2.0 - 1.0


def ycbcr_to_rgb(x):
    x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    y, cb, cr = x01[:, 0:1], x01[:, 1:2] - 0.5, x01[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb01 = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb01 * 2.0 - 1.0, -1.0, 1.0)


# ============================================================================
# Model Components
# ============================================================================

class LearnableSubbandWeights(nn.Module):
    def __init__(self, init_wLL=0.35, init_wLH=0.14, init_wHL=0.14, init_wHH=0.06, init_wLL2=0.20):
        super(LearnableSubbandWeights, self).__init__()
        self.log_wLL = nn.Parameter(torch.log(torch.tensor(init_wLL)))
        self.log_wLH = nn.Parameter(torch.log(torch.tensor(init_wLH)))
        self.log_wHL = nn.Parameter(torch.log(torch.tensor(init_wHL)))
        self.log_wHH = nn.Parameter(torch.log(torch.tensor(init_wHH)))
        self.log_wLL2 = nn.Parameter(torch.log(torch.tensor(init_wLL2)))
    
    def get_weights(self):
        wLL = torch.exp(self.log_wLL)
        wLH = torch.exp(self.log_wLH)
        wHL = torch.exp(self.log_wHL)
        wHH = torch.exp(self.log_wHH)
        wLL2 = torch.exp(self.log_wLL2)
        return torch.stack([wLL, wLH, wHL, wHH]), wLL2


class LearnableYCbCrScaling(nn.Module):
    def __init__(self, init_kY=0.04, init_kC=0.10):
        super(LearnableYCbCrScaling, self).__init__()
        self.log_kY = nn.Parameter(torch.log(torch.tensor(init_kY)))
        self.log_kC = nn.Parameter(torch.log(torch.tensor(init_kC)))
    
    def get_scales(self):
        kY = torch.exp(self.log_kY)
        kC = torch.exp(self.log_kC)
        return kY, kC


class ActNorm(nn.Module):
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
                std = torch.std(x, dim=[0, 2, 3], keepdim=True)
                self.scale.data.copy_(1.0 / (std + 1e-6))
                self.bias.data.copy_(-mean * self.scale.data)
                self.initialized.fill_(1)
        y = x * self.scale + self.bias
        log_det = torch.sum(torch.log(torch.abs(self.scale))) * x.shape[2] * x.shape[3]
        return y, log_det

    def inverse(self, y):
        x = (y - self.bias) / self.scale
        return x


class AffineCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels=64, cond_channels=0):
        super(AffineCouplingLayer, self).__init__()
        assert channels % 2 == 0
        half_channels = channels // 2
        in_ch = half_channels + cond_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv2d(hidden_channels, half_channels * 2, kernel_size=3, padding=1),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
        self.gamma_raw = nn.Parameter(torch.zeros(1, half_channels, 1, 1))

    def forward(self, x, cond=None):
        x1, x2 = x.chunk(2, dim=1)
        if cond is not None:
            x1_in = torch.cat([x1, cond], dim=1)
        else:
            x1_in = x1
        s_t = self.net(x1_in)
        s, t = s_t.chunk(2, dim=1)
        s = torch.tanh(s)
        gamma = F.softplus(self.gamma_raw)
        y1 = x1
        y2 = torch.exp(s) * x2 + t + gamma * x2
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y, cond=None):
        y1, y2 = y.chunk(2, dim=1)
        if cond is not None:
            y1_in = torch.cat([y1, cond], dim=1)
        else:
            y1_in = y1
        s_t = self.net(y1_in)
        s, t = s_t.chunk(2, dim=1)
        s = torch.tanh(s)
        gamma = F.softplus(self.gamma_raw)
        x1 = y1
        denom = torch.exp(s) + gamma + 1e-6
        x2 = (y2 - t) / denom
        return torch.cat([x1, x2], dim=1)


class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        self.channels = channels
        w_init = np.random.randn(channels, channels)
        q, _ = np.linalg.qr(w_init)
        w_init = torch.from_numpy(q.astype(np.float32))
        self.weight = nn.Parameter(w_init)

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        y_flat = torch.matmul(self.weight, x_flat)
        y = y_flat.view(b, c, h, w)
        log_det = h * w * torch.log(torch.abs(torch.det(self.weight)))
        return y, log_det

    def inverse(self, y):
        b, c, h, w = y.size()
        w_inv = torch.inverse(self.weight)
        y_flat = y.view(b, c, -1)
        x_flat = torch.matmul(w_inv, y_flat)
        x = x_flat.view(b, c, h, w)
        return x


# StegoWaveFlow Block
class StegoWaveFlowBlock(nn.Module):
    def __init__(self, channels, hidden_channels=64, cond_channels=0):
        super(StegoWaveFlowBlock, self).__init__()
        assert channels % 2 == 0
        self.actnorm = ActNorm(channels)
        self.inv_conv = Invertible1x1Conv(channels)
        self.affine_coupling = AffineCouplingLayer(channels, hidden_channels, cond_channels)

    def forward(self, x, cond=None):
        x, log_det1 = self.actnorm(x)
        x, log_det2 = self.inv_conv(x)
        x = self.affine_coupling(x, cond)
        log_det = log_det1 + log_det2
        return x, log_det

    def inverse(self, y, cond=None):
        y = self.affine_coupling.inverse(y, cond)
        y = self.inv_conv.inverse(y)
        y = self.actnorm.inverse(y)
        return y


# StegoWaveFlow with ILWT
class StegoWaveFlow(nn.Module):
    def __init__(self, channels=6, num_blocks=3, hidden_channels=48, transform_type: str = "ilwt53"):
        super(StegoWaveFlow, self).__init__()
        if transform_type == "ilwt53":
            self.ilwt = ILWT53_2D(channels)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
        self.inn_channels = channels * 4
        cond_ch = 16
        self.cond_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(StegoWaveFlowBlock(self.inn_channels, hidden_channels, cond_channels=cond_ch))
        self.subband_weights = LearnableSubbandWeights()
        self.ycbcr_scaling = LearnableYCbCrScaling()

    def forward(self, x):
        cover = x[:, :3, :, :]
        cond = self.cond_net(cover)
        x_freq = self.ilwt(x)
        cond_resized = F.interpolate(cond, size=(x_freq.shape[2], x_freq.shape[3]), mode="bilinear", align_corners=False)
        log_det_total = 0
        z = x_freq
        for block in self.blocks:
            z, log_det = block(z, cond_resized)
            log_det_total += log_det
        residual_spatial = self.ilwt.inverse(z)
        cover_ycc = rgb_to_ycbcr(cover)
        resid_ycc = rgb_to_ycbcr(residual_spatial[:, :3, :, :])
        kY, kC = self.ycbcr_scaling.get_scales()
        scale = torch.stack([kY, kC, kC]).view(1, 3, 1, 1).to(cover.device)
        stego_ycc = cover_ycc + torch.tanh(resid_ycc) * scale
        stego_rgb = ycbcr_to_rgb(stego_ycc)
        stego_spatial = x.clone()
        stego_spatial[:, :3, :, :] = stego_rgb
        stego_spatial[:, 3:, :, :] = residual_spatial[:, 3:, :, :]
        return stego_spatial, log_det_total

    def inverse(self, z):
        z_freq = self.ilwt(z)
        cond = self.cond_net(z[:, :3, :, :])
        cond_resized = F.interpolate(cond, size=(z_freq.shape[2], z_freq.shape[3]), mode="bilinear", align_corners=False)
        x_freq = z_freq
        for i in range(len(self.blocks) - 1, -1, -1):
            x_freq = self.blocks[i].inverse(x_freq, cond_resized)
        residual_spatial = self.ilwt.inverse(x_freq)
        z_ycc = rgb_to_ycbcr(z[:, :3, :, :])
        resid_ycc = rgb_to_ycbcr(residual_spatial[:, :3, :, :])
        kY, kC = self.ycbcr_scaling.get_scales()
        scale = torch.stack([kY, kC, kC]).view(1, 3, 1, 1).to(z.device)
        host_ycc = z_ycc - torch.tanh(resid_ycc) * scale
        host_rgb = ycbcr_to_rgb(host_ycc)
        x_spatial = z.clone()
        x_spatial[:, :3, :, :] = host_rgb
        x_spatial[:, 3:, :, :] = residual_spatial[:, 3:, :, :]
        return x_spatial


@runtime_checkable
class SteganographyModel(Protocol):
    """Protocol for model checking."""
    def forward(self, x: Any) -> tuple:
        ...
    def inverse(self, z: Any) -> Any:
        ...


def create_model(config: ModelConfig) -> Any:
    """Create a steganography model from configuration."""
    model = StegoWaveFlow(
        channels=config.channels,
        num_blocks=config.num_blocks,
        hidden_channels=config.hidden_channels,
        transform_type=config.transform_type
    )
    return model


def get_model_info(config: ModelConfig) -> dict:
    """Get model information."""
    return {
        'model_path': config.path,
        'architecture': 'StegoWaveFlow',
        'num_blocks': config.num_blocks,
        'hidden_channels': config.hidden_channels,
        'channels': config.channels,
        'transform_type': config.transform_type,
        'input_size': config.size,
        'inn_channels': config.channels * 4,
    }

MODEL_REGISTRY = {
    'stegowaveflow_ilwt': {
        'factory': create_model,
        'description': 'StegoWaveFlow with Integer Lifting Wavelet Transform',
    }
}
