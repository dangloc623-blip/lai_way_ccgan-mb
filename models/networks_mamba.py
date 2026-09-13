"""
networks_mamba.py
=================
Lightweight Mamba-CycleGAN network components:

  - SelectiveScan2D     : Pure-PyTorch 2-direction selective scan (S6 core)
  - VSSBlock            : Vision State Space Block (Mamba 2-D)
  - DWSConvBlock        : Depthwise Separable Conv block
  - MambaUNetGenerator  : Hybrid U-Net encoder/decoder + Mamba bottleneck
  - LitePatchGANDiscriminator : PatchGAN with DWSConv + reduced channels
  - SSIMLoss            : Structural Similarity loss
  - define_G_mamba      : Generator factory
  - define_D_lite       : Discriminator factory
"""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from .networks import get_norm_layer, init_net


# =============================================================================
# 1. Selective Scan — pure PyTorch, 2-direction cross-scan
# =============================================================================

class SelectiveScan2D(nn.Module):
    """
    Simplified Selective Scan (S6) for 2-D images.
    Scans in 2 directions:
      - Direction 0: raster    (top-left → bottom-right)
      - Direction 1: anti-raster (bottom-right → top-left)

    Input : (B, C, H, W)
    Output: (B, C, H, W)

    The SSM is parameterised as:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t + D * x_t        (D is a skip/residual scalar)

    A is a diagonal matrix (represented as a vector of size d_state),
    so the recurrence is element-wise and fully parallelisable per head.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3,
                 expand: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv on the expanded feature (local context before scan)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM parameters — one set per scan direction
        n_dirs = 2
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)   # dt, B, C from x
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)                  # dt projection

        # A: log-diagonal, shape (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Per-direction projection (weight-sharing across H×W, separate per dir)
        self.dir_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.d_inner, bias=False)
            for _ in range(n_dirs)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Core SSM scan (recurrent, runs over sequence length L)
    # ------------------------------------------------------------------
    def _ssm_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        u : (B, L, d_inner)
        returns y : (B, L, d_inner)
        """
        # [PATCH v2] CHUNKED PARALLEL SCAN (thay vong lap tung-buoc-mot bang
        # vong lap qua CHUNK, moi chunk tinh song song noi bo). Cong thuc toan
        # hoc CHINH XAC giong ban tuan tu (da kiem chung bang so hoc rieng).
        orig_dtype = u.dtype
        with torch.cuda.amp.autocast(enabled=False):
            u = u.float()
            B, L, D = u.shape
            d_state = self.d_state
            _SCAN_CHUNK = 64
            Csz = min(_SCAN_CHUNK, L)

            x_dbc = self.x_proj(u)                          # (B, L, d_state*2+1)
            dt_raw = x_dbc[..., :1]                          # (B, L, 1)
            B_ssm = x_dbc[..., 1:d_state + 1]               # (B, L, d_state)
            C_ssm = x_dbc[..., d_state + 1:]                # (B, L, d_state)

            dt = F.softplus(self.dt_proj(dt_raw))            # (B, L, d_inner)
            A = -torch.exp(self.A_log)                       # (d_inner, d_state)

            def _process_chunk(dt_c, Bs_c, Cs_c, u_c, H_in):
                # dt_c,u_c: (B,c,D) ; Bs_c,Cs_c: (B,c,S) ; H_in: (B,D,S)
                c = dt_c.shape[1]
                # [FIX buoc 6] logA_c = log(dA_c) = log(exp(dt*A)) = dt*A THANG,
                # khong di vong qua exp() roi log() lai (di vong bi underflow
                # ve log(0)=-inf khi dt*A rat am).
                logA_c = dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)   # (B,c,D,S) = log(dA_c)
                cumlog = torch.cumsum(logA_c, dim=1)               # (B,c,D,S), <= 0 luon dung

                h_from_H = torch.exp(cumlog) * H_in.unsqueeze(1)   # dong gop tu carry chunk truoc

                dB_c = dt_c.unsqueeze(-1) * Bs_c.unsqueeze(2)       # (B,c,D,S)
                v = dB_c * u_c.unsqueeze(-1)                          # (B,c,D,S)

                cumlog_i = cumlog.unsqueeze(2)                        # (B,c,1,D,S)
                cumlog_k = cumlog.unsqueeze(1)                        # (B,1,c,D,S)
                # [FIX buoc 6] clamp <=0 TRUOC exp: cac cap se bi mask (k>i) co
                # hieu duong/lon, exp truc tiep se TRAN SO (+inf) roi inf*0=NaN
                # khi nhan voi mask ben duoi. Clamp truoc dam bao decay_ik luon
                # huu han trong [0,1], mask sau do van loai dung cac cap khong
                # hop le (gia tri clamp cua chung khong quan trong vi se bi *0).
                diff = torch.clamp(cumlog_i - cumlog_k, max=0.0)
                decay_ik = torch.exp(diff)                              # (B,c,c,D,S), luon trong [0,1]
                mask = torch.tril(torch.ones(c, c, device=u.device, dtype=u.dtype))
                decay_ik = decay_ik * mask.view(1, c, c, 1, 1)          # chi giu k<=i

                # [PATCH buoc 7] gop (D,S) vao truc batch de dung bmm() (GEMM toi uu)
                # thay vi einsum 5D (thuong cham hon nhieu tren GPU).
                _Bc, _c1, _c2, _Dc, _Sc = decay_ik.shape
                _decay_bmm = decay_ik.permute(0, 3, 4, 1, 2).reshape(_Bc * _Dc * _Sc, _c1, _c2)
                _v_bmm = v.permute(0, 2, 3, 1).reshape(_Bc * _Dc * _Sc, _c2, 1)
                h_intra = torch.bmm(_decay_bmm, _v_bmm).reshape(_Bc, _Dc, _Sc, _c1).permute(0, 3, 1, 2)  # dong gop noi bo chunk
                h_chunk = h_from_H + h_intra                              # (B,c,D,S) = h tai tung buoc

                y_chunk = (h_chunk * Cs_c.unsqueeze(2)).sum(-1)           # (B,c,D)
                H_out = h_chunk[:, -1]                                      # carry cho chunk sau
                return H_out, y_chunk

            H = torch.zeros(B, D, d_state, device=u.device, dtype=u.dtype)
            outs = []
            use_ckpt = self.training and u.requires_grad
            for start in range(0, L, Csz):
                end = min(start + Csz, L)
                dt_c, Bs_c, Cs_c, u_c = dt[:, start:end], B_ssm[:, start:end], C_ssm[:, start:end], u[:, start:end]
                if use_ckpt:
                    # checkpoint TUNG CHUNK (khong phai ca ham) -- de luc backward chi 1
                    # chunk giu do thi O(C^2) cung luc, tranh OOM.
                    H, y_chunk = _grad_checkpoint(_process_chunk, dt_c, Bs_c, Cs_c, u_c, H,
                                                   use_reentrant=False)
                else:
                    H, y_chunk = _process_chunk(dt_c, Bs_c, Cs_c, u_c, H)
                outs.append(y_chunk)

            y = torch.cat(outs, dim=1)                        # (B, L, D)
            y = y + u * self.D.unsqueeze(0).unsqueeze(0)      # skip connection
        return y.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)"""
        B, C, H, W = x.shape
        identity = x

        # Flatten spatial → sequence
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, L, C)

        # Input projection → (B, L, d_inner*2) split to z and x2
        xz = self.in_proj(x_flat)
        x2, z = xz.chunk(2, dim=-1)                            # (B, L, d_inner) each

        # Local conv (along L dim)
        x2_conv = self.conv1d(x2.permute(0, 2, 1))[..., :H * W].permute(0, 2, 1)
        x2_conv = F.silu(x2_conv)

        # 2-direction scan
        outs = []
        for d, proj in enumerate(self.dir_proj):
            seq = proj(x2_conv)
            if d == 1:                  # reverse direction
                seq = seq.flip(1)
            # Checkpoint gio nam BEN TRONG _ssm_scan (theo tung chunk), khong con
            # checkpoint ca ham o day nua (xem [PATCH v2] trong _ssm_scan).
            if self.training and seq.requires_grad:
                # gradient checkpointing: khong luu activation trung gian cua
                # vong lap 4096 buoc, tinh lai luc backward -> giam manh VRAM
                y = _grad_checkpoint(self._ssm_scan, seq, use_reentrant=False)
            else:
                y = self._ssm_scan(seq)    # (B, L, d_inner)
            if d == 1:
                y = y.flip(1)
            outs.append(y)

        y = sum(outs) / len(outs)      # average across directions

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)           # (B, L, C)
        y = self.norm(y)

        # Reshape back to spatial
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)         # (B, C, H, W)
        return y + identity                                     # residual


# =============================================================================
# 2. VSSBlock — Vision State Space Block (Mamba bottleneck unit)
# =============================================================================

class VSSBlock(nn.Module):
    """
    VSS Block = LayerNorm + SelectiveScan2D + Feed-Forward (DWSConv)
    Following the VMamba design: norm → scan → norm → FFN
    """

    def __init__(self, dim: int, d_state: int = 16, expand: float = 2.0,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)   # equivalent to LayerNorm on channels
        self.scan = SelectiveScan2D(dim, d_state=d_state, expand=expand)

        self.norm2 = nn.GroupNorm(1, dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),  # DW
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scan branch
        x = x + self.scan(self.norm1(x))
        # FFN branch
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# 3. Depthwise Separable Conv Block
# =============================================================================

class DWSConvBlock(nn.Module):
    """
    Depthwise Separable Conv block:
      DW Conv → PW Conv → Norm → Activation

    Reduces parameters ~8× vs standard Conv for the same channel config.
    """

    def __init__(self, in_c: int, out_c: int, kernel: int = 3,
                 stride: int = 1, norm_layer=nn.InstanceNorm2d,
                 use_bias: bool = False, activation: bool = True):
        super().__init__()
        pad = kernel // 2
        layers = [
            # Depthwise
            nn.Conv2d(in_c, in_c, kernel, stride=stride,
                      padding=pad, groups=in_c, bias=use_bias),
            # Pointwise
            nn.Conv2d(in_c, out_c, 1, bias=use_bias),
            norm_layer(out_c),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# =============================================================================
# 4. MambaUNetGenerator
# =============================================================================

class MambaUNetGenerator(nn.Module):
    """
    Hybrid U-Net Generator with Mamba bottleneck.

    Architecture
    ------------
    Encoder  (3 × DWSConvBlock, stride=2):
        in_nc  →  ngf  →  ngf*2  →  ngf*4   (H → H/8)

    Bottleneck  (n_mamba × VSSBlock):
        ngf*4 at spatial resolution H/8 × W/8

    Decoder  (3 × Bilinear-Upsample + DWSConvBlock + skip concat):
        ngf*4 → ngf*2 → ngf → ngf

    Head  (Conv 7×7 + Tanh):
        ngf → out_nc
    """

    def __init__(self, input_nc: int, output_nc: int, ngf: int = 64,
                 norm_layer=nn.InstanceNorm2d, n_mamba: int = 3,
                 d_state: int = 16):
        super().__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ---- Encoder ----
        # Initial 7×7 conv (reflect-padded, as in ResNet CycleGAN)
        self.enc0_pad = nn.ReflectionPad2d(3)
        self.enc0_conv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DWSConvBlock(ngf,     ngf * 2, 3, stride=2,
                                 norm_layer=norm_layer, use_bias=use_bias)
        self.enc2 = DWSConvBlock(ngf * 2, ngf * 4, 3, stride=2,
                                 norm_layer=norm_layer, use_bias=use_bias)

        # ---- Bottleneck (Mamba) ----
        self.bottleneck = nn.Sequential(
            *[VSSBlock(ngf * 4, d_state=d_state) for _ in range(n_mamba)]
        )

        # ---- Decoder ----
        # Each decoder step: upsample × 2, then DWSConvBlock on concat feat
        # After skip-concat channels double; DWSConv reduces them back.
        self.dec2 = DWSConvBlock(ngf * 4 + ngf * 4, ngf * 2, 3,
                                 norm_layer=norm_layer, use_bias=use_bias)
        self.dec1 = DWSConvBlock(ngf * 2 + ngf * 2, ngf,     3,
                                 norm_layer=norm_layer, use_bias=use_bias)
        self.dec0 = DWSConvBlock(ngf     + ngf,      ngf,     3,
                                 norm_layer=norm_layer, use_bias=use_bias)

        # ---- Output head ----
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0_conv(self.enc0_pad(x))   # (B, ngf,   H,   W)
        e1 = self.enc1(e0)                       # (B, ngf*2, H/2, W/2)
        e2 = self.enc2(e1)                       # (B, ngf*4, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(e2)                  # (B, ngf*4, H/4, W/4)

        # Decoder with skip connections -- [PATCH] noi (concat) TRUOC khi kich
        # thuoc con khop voi tang encoder tuong ung, upsample SAU do cho tang
        # ke tiep (thay vi upsample truoc roi noi lech kich thuoc nhu ban goc).
        d2 = self.dec2(torch.cat([b, e2], dim=1))     # (B, ngf*2, H/4, W/4)

        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))    # (B, ngf,   H/2, W/2)

        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        d0 = self.dec0(torch.cat([d1, e0], dim=1))    # (B, ngf,   H,   W)

        return self.head(d0)                          # (B, out_nc, H, W)


# =============================================================================
# 5. LitePatchGANDiscriminator
# =============================================================================

class LitePatchGANDiscriminator(nn.Module):
    """
    Lite PatchGAN Discriminator.
    Identical receptive field to vanilla PatchGAN (70×70) but uses
    Depthwise Separable Convolutions and ndf=32 to cut parameters ~75%.
    """

    def __init__(self, input_nc: int, ndf: int = 32,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        def dws_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 4, stride=stride,
                          padding=1, groups=in_c, bias=use_bias),
                nn.Conv2d(in_c, out_c, 1, bias=use_bias),
                norm_layer(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Layer 0: plain Conv (no norm on first layer — standard PatchGAN)
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer1 = dws_block(ndf,        ndf * 2, stride=2)
        self.layer2 = dws_block(ndf * 2,    ndf * 4, stride=2)
        self.layer3 = dws_block(ndf * 4,    ndf * 8, stride=1)

        # Final 1-channel prediction map
        self.out = nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.out(x)


# =============================================================================
# 6. SSIM Loss
# =============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    Loss = 1 - SSIM(pred, target), averaged over batch.

    Window size 11, sigma 1.5 — standard settings from Wang et al. 2004.
    Works for single-channel or multi-channel images (averaged over channels).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 2.0):
        """
        data_range: pixel value range.
            Use 2.0 when images are in [-1, 1] (CycleGAN default).
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        # Build 2-D Gaussian kernel
        kernel = self._gaussian_kernel(window_size, sigma)
        # Shape: (1, 1, window_size, window_size) — will be expanded per channel
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return torch.outer(g, g)

    def _ssim_single(self, x: torch.Tensor, y: torch.Tensor,
                     num_channels: int) -> torch.Tensor:
        """Compute SSIM for a pair of single-or-multi-channel images."""
        kernel = self.kernel.expand(num_channels, 1, -1, -1)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, kernel, padding=pad, groups=num_channels)
        mu_y = F.conv2d(y, kernel, padding=pad, groups=num_channels)

        mu_x2  = mu_x * mu_x
        mu_y2  = mu_y * mu_y
        mu_xy  = mu_x * mu_y

        sigma_x2  = F.conv2d(x * x, kernel, padding=pad, groups=num_channels) - mu_x2
        sigma_y2  = F.conv2d(y * y, kernel, padding=pad, groups=num_channels) - mu_y2
        sigma_xy  = F.conv2d(x * y, kernel, padding=pad, groups=num_channels) - mu_xy

        numerator   = (2 * mu_xy   + self.C1) * (2 * sigma_xy  + self.C2)
        denominator = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)

        return (numerator / denominator).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, C, H, W) — values in [-1, 1]
        Returns scalar loss (1 - SSIM).
        """
        nc = pred.shape[1]
        ssim_val = self._ssim_single(pred, target, nc)
        return 1.0 - ssim_val


# =============================================================================
# 7. Factory functions
# =============================================================================

def define_G_mamba(input_nc: int, output_nc: int, ngf: int,
                   netG: str, norm: str = 'instance',
                   init_type: str = 'normal', init_gain: float = 0.02,
                   n_mamba: int = 3, d_state: int = 16):
    """
    Create a Mamba-based Generator.

    netG choices
    ------------
    'mamba_unet'  : MambaUNetGenerator (recommended)

    Parameters mirror define_G() in networks.py for drop-in compatibility.
    """
    norm_layer = get_norm_layer(norm)

    if netG == 'mamba_unet':
        net = MambaUNetGenerator(input_nc, output_nc, ngf,
                                 norm_layer=norm_layer,
                                 n_mamba=n_mamba,
                                 d_state=d_state)
    else:
        raise NotImplementedError(f'Generator [{netG}] not recognised in networks_mamba.')

    return init_net(net, init_type, init_gain)


def define_D_lite(input_nc: int, ndf: int,
                  netD: str, norm: str = 'instance',
                  init_type: str = 'normal', init_gain: float = 0.02):
    """
    Create a Lite-PatchGAN Discriminator.

    netD choices
    ------------
    'lite_patch'  : LitePatchGANDiscriminator (recommended)
    """
    norm_layer = get_norm_layer(norm)

    if netD == 'lite_patch':
        net = LitePatchGANDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f'Discriminator [{netD}] not recognised in networks_mamba.')

    return init_net(net, init_type, init_gain)

