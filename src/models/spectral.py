"""2D Fast Fourier Transform spectral decomposition with numerically stable autograd."""

import torch
import torch.nn as nn


class RealFFT2DModule(nn.Module):
    """
    2D FFT spectral decomposition into log-magnitude and normalized phase components.
    Uses sub-epsilon masked float32 atan2 autograd to guarantee bounded, finite gradients
    near zero frequency bins without exploding to ±10^11 or creating NaN/Inf values.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [B, C, H, W]  (e.g., C=10 noise maps)
        Output: Spectral features [B, 2*C, H, W] (C_out=20 concatenated mag & phase)
        """
        device_type = x.device.type if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            fft = torch.fft.fft2(x_fp32, norm="ortho")
            fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

            abs_fft = torch.abs(fft_shift)
            mag = torch.log1p(torch.clamp(abs_fft, min=1e-7))

            mask = abs_fft < self.eps
            safe_real = torch.where(mask, torch.ones_like(fft_shift.real), fft_shift.real)
            safe_imag = torch.where(mask, torch.zeros_like(fft_shift.imag), fft_shift.imag)
            phase_raw = torch.atan2(safe_imag, safe_real) / torch.pi
            phase = torch.where(mask, torch.zeros_like(phase_raw), phase_raw)

            mag = torch.nan_to_num(mag, nan=0.0, posinf=10.0, neginf=-10.0)
            phase = torch.nan_to_num(phase, nan=0.0, posinf=1.0, neginf=-1.0)
            out_fp32 = torch.cat([mag, phase], dim=1)

        return out_fp32.to(dtype=x.dtype, device=x.device)
