"""
slice25d_dataset.py
===================
Dataset for 2.5D medical image translation (CycleGAN / Mamba-CycleGAN).

What is 2.5D?
-------------
Instead of training on isolated 2-D slices, we stack **n_slices** consecutive
axial (or sagittal/coronal) slices into a single multi-channel tensor.
This gives the 2-D network spatial context along the third dimension, at a
fraction of the memory cost of full 3-D convolutions.

Supported input formats
-----------------------
PNG / JPG / BMP  (grayscale or RGB; converted to grayscale internally)
NPY              (.npy files containing (H, W) float32 arrays, pre-normalised
                  to [0, 1] or any range — auto-rescaled to [-1, 1])

Directory structure
-------------------
    <dataroot>/
        trainA/   ← domain A slices, named so that alphabetical order == slice order
        trainB/   ← domain B slices
        testA/
        testB/

File naming convention
----------------------
Files must sort correctly to represent slice order.  A simple zero-padded
integer suffix works well, e.g.:
    patient01_0000.png, patient01_0001.png, ..., patient01_0199.png

Each __getitem__ returns a 2.5D window centred on index i:
    channels = [slice_{i - n//2}, ..., slice_i, ..., slice_{i + n//2}]
Boundary slices are handled by padding (replicate the first / last slice).

Usage example
-------------
    python train.py \\
        --model mamba_cycle_gan \\
        --dataset_mode slice25d \\
        --dataroot /data/ct_dataset \\
        --n_slices 3 \\
        --input_nc 3  --output_nc 1 \\
        --crop_size 256 --load_size 286
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from data.base_dataset import BaseDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def _sorted_files(folder: str) -> list:
    """Return sorted list of image / npy files in a folder."""
    p = Path(folder)
    files = sorted([
        str(f) for f in p.iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS or f.suffix.lower() == '.npy'
    ])
    return files


def _load_slice(path: str) -> np.ndarray:
    """
    Load one slice and return a float32 numpy array of shape (H, W)
    with values in [0, 1].
    """
    if path.endswith('.npy'):
        arr = np.load(path).astype(np.float32)
        # normalise to [0, 1] if needed
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr
    else:
        img = Image.open(path).convert('L')   # grayscale
        return np.asarray(img, dtype=np.float32) / 255.0


def _to_tensor_norm(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W) float32 [0,1] array → (1, H, W) tensor in [-1, 1]."""
    t = torch.from_numpy(arr).unsqueeze(0)   # (1, H, W)
    return t * 2.0 - 1.0                     # [0,1] → [-1,1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Slice25dDataset(BaseDataset):
    """
    Unpaired 2.5D slice dataset.

    Returns
    -------
    dict with keys:
        'A'       : (n_slices, H, W) tensor — 2.5D input (domain A)
        'B'       : (1, H, W) tensor        — target slice (domain B)
        'A_paths' : str path of the centre slice in A
        'B_paths' : str path of the centre slice in B
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_slices', type=int, default=3,
                            help='Number of consecutive slices stacked as '
                                 'input channels (2.5D). Must be odd. '
                                 'Must equal --input_nc.')
        parser.set_defaults(input_nc=3, output_nc=1)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.n_slices = opt.n_slices
        self.phase    = opt.phase          # 'train' / 'test' / 'val'
        self.is_train = (self.phase == 'train')

        # Locate slice files
        dir_A = Path(opt.dataroot) / f'{self.phase}A'
        dir_B = Path(opt.dataroot) / f'{self.phase}B'

        if not dir_A.exists():
            raise FileNotFoundError(f'Domain-A folder not found: {dir_A}')
        if not dir_B.exists():
            raise FileNotFoundError(f'Domain-B folder not found: {dir_B}')

        self.files_A = _sorted_files(str(dir_A))
        self.files_B = _sorted_files(str(dir_B))

        if len(self.files_A) == 0:
            raise RuntimeError(f'No image/npy files found in {dir_A}')
        if len(self.files_B) == 0:
            raise RuntimeError(f'No image/npy files found in {dir_B}')

        self.size_A = len(self.files_A)
        self.size_B = len(self.files_B)

        # Spatial transform parameters
        self.crop_size = opt.crop_size
        self.load_size = opt.load_size
        self.no_flip   = opt.no_flip

    # -----------------------------------------------------------------------
    def __len__(self):
        return max(self.size_A, self.size_B)

    # -----------------------------------------------------------------------
    def _load_25d_window(self, files: list, centre_idx: int) -> torch.Tensor:
        """
        Load n_slices slices centred on centre_idx.
        Returns tensor of shape (n_slices, H, W) in [-1, 1].
        Boundary slices are replicated.
        """
        half = self.n_slices // 2
        n    = len(files)
        slices = []
        for offset in range(-half, half + 1):
            idx   = min(max(centre_idx + offset, 0), n - 1)  # clamp
            arr   = _load_slice(files[idx])
            slices.append(arr)
        # Stack → (n_slices, H, W)
        volume = np.stack(slices, axis=0)
        # To tensor: (n_slices, H, W) in [-1, 1]
        tensor = torch.from_numpy(volume).float() * 2.0 - 1.0
        return tensor

    def _load_single(self, files: list, idx: int) -> torch.Tensor:
        """Load one slice as (1, H, W) tensor in [-1, 1]."""
        arr = _load_slice(files[idx])
        return _to_tensor_norm(arr)

    # -----------------------------------------------------------------------
    def _shared_transform(self, *tensors):
        """
        Apply the same random crop + flip to all tensors.
        Each tensor: (C, H, W).
        Returns list of transformed tensors.
        """
        # 1. Resize
        resized = []
        for t in tensors:
            # torchvision Resize expects (C, H, W)
            t = T.Resize((self.load_size, self.load_size),
                         interpolation=T.InterpolationMode.BICUBIC,
                         antialias=True)(t)
            resized.append(t)

        # 2. Random crop (same position for A and B)
        i, j, h, w = T.RandomCrop.get_params(
            resized[0], output_size=(self.crop_size, self.crop_size)
        )
        cropped = [TF.crop(t, i, j, h, w) for t in resized]

        # 3. Random horizontal flip
        if self.is_train and not self.no_flip and random.random() > 0.5:
            cropped = [TF.hflip(t) for t in cropped]

        return cropped

    # -----------------------------------------------------------------------
    def __getitem__(self, index: int):
        idx_A = index % self.size_A
        idx_B = random.randint(0, self.size_B - 1)  # unpaired (CycleGAN)

        # Load tensors
        tensor_A = self._load_25d_window(self.files_A, idx_A)   # (n_slices, H, W)
        tensor_B = self._load_single(self.files_B, idx_B)        # (1, H, W)

        if self.is_train:
            tensor_A, tensor_B = self._shared_transform(tensor_A, tensor_B)
        else:
            # Test: just resize, no random crop/flip
            tensor_A = T.Resize((self.crop_size, self.crop_size),
                                 interpolation=T.InterpolationMode.BICUBIC,
                                 antialias=True)(tensor_A)
            tensor_B = T.Resize((self.crop_size, self.crop_size),
                                 interpolation=T.InterpolationMode.BICUBIC,
                                 antialias=True)(tensor_B)

        return {
            'A':       tensor_A,
            'B':       tensor_B,
            'A_paths': self.files_A[idx_A],
            'B_paths': self.files_B[idx_B],
        }

