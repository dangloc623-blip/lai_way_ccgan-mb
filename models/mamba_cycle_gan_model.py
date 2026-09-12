"""
mamba_cycle_gan_model.py
========================
MambaCycleGAN: CycleGAN with Mamba-UNet generator, Lite-PatchGAN discriminator,
and SSIM-augmented losses for medical image translation.

Key differences vs. standard CycleGANModel
-------------------------------------------
- Generator  : MambaUNetGenerator  (CNN encoder/decoder + Mamba bottleneck)
- Discriminator : LitePatchGANDiscriminator (DWSConv, ndf=32)
- Extra loss  : SSIM loss on forward & backward cycle reconstructions
- 2.5D support: input_nc = n_slices (default 3), output_nc = 1

Usage
-----
    python train.py \\
        --model mamba_cycle_gan \\
        --netG mamba_unet \\
        --netD lite_patch \\
        --dataset_mode slice25d \\
        --input_nc 3 --output_nc 1 \\
        --n_mamba 3 --n_slices 3 \\
        --lambda_ssim 1.0 \\
        --ngf 64 --ndf 32
"""

import torch
import itertools

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks_mamba import (
    define_G_mamba,
    define_D_lite,
    SSIMLoss,
)


class MambaCycleGANModel(BaseModel):
    """
    Mamba-CycleGAN model for unpaired medical image translation.

    Generators  : G_A (A→B), G_B (B→A) — MambaUNetGenerator
    Discriminators: D_A (real B vs fake B), D_B (real A vs fake A) — LitePatchGAN

    Loss breakdown
    --------------
    L_G  = L_GAN_A + L_GAN_B
         + λ_A  * L_cycle_A  + λ_B  * L_cycle_B
         + λ_idt * (λ_B * L_idt_A + λ_A * L_idt_B)
         + λ_ssim * (L_ssim_A + L_ssim_B)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add Mamba-CycleGAN-specific arguments and override defaults."""
        # Sensible defaults for medical imaging
        parser.set_defaults(
            no_dropout=True,
            norm='instance',
            netG='mamba_unet',
            netD='lite_patch',
            ngf=64,
            ndf=32,
            input_nc=3,   # 3 slices (2.5D)
            output_nc=1,  # CT grayscale output
            dataset_mode='slice25d',
        )

        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='Weight for cycle loss A→B→A')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='Weight for cycle loss B→A→B')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='Weight factor for identity mapping loss. '
                                     '0 = disabled.')
            parser.add_argument('--lambda_ssim', type=float, default=1.0,
                                help='Weight for SSIM loss on cycle reconstructions. '
                                     '0 = disabled.')

        # Mamba architecture args (used at both train and test time)
        parser.add_argument('--n_mamba', type=int, default=3,
                            help='Number of VSSBlocks in the Mamba bottleneck '
                                 '(2–4 recommended).')
        parser.add_argument('--d_state', type=int, default=16,
                            help='SSM state dimension inside each VSSBlock.')
        parser.add_argument('--n_slices', type=int, default=3,
                            help='Number of consecutive slices stacked as channels '
                                 '(2.5D input). Must equal input_nc.')
        return parser

    # ------------------------------------------------------------------
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # ---------- losses to log ----------
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'ssim_A',
                           'D_B', 'G_B', 'cycle_B', 'idt_B', 'ssim_B']

        # ---------- images to visualise ----------
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B

        # ---------- networks ----------
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # Generators
        self.netG_A = define_G_mamba(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            netG=opt.netG,
            norm=opt.norm,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            n_mamba=opt.n_mamba,
            d_state=opt.d_state,
        )
        self.netG_B = define_G_mamba(
            input_nc=opt.output_nc,
            output_nc=opt.input_nc,
            ngf=opt.ngf,
            netG=opt.netG,
            norm=opt.norm,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            n_mamba=opt.n_mamba,
            d_state=opt.d_state,
        )

        if self.isTrain:
            # Discriminators
            self.netD_A = define_D_lite(
                input_nc=opt.output_nc,
                ndf=opt.ndf,
                netD=opt.netD,
                norm=opt.norm,
                init_type=opt.init_type,
                init_gain=opt.init_gain,
            )
            self.netD_B = define_D_lite(
                input_nc=opt.input_nc,
                ndf=opt.ndf,
                netD=opt.netD,
                norm=opt.norm,
                init_type=opt.init_type,
                init_gain=opt.init_gain,
            )

            # Guard: identity loss requires same channel count
            if opt.lambda_identity > 0.0:
                assert opt.input_nc == opt.output_nc, (
                    'Identity loss requires input_nc == output_nc. '
                    'Set --lambda_identity 0 for 2.5D (in_nc=3, out_nc=1).'
                )

            # Image buffers (replay buffer)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # ---------- loss functions ----------
            self.criterionGAN   = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt   = torch.nn.L1Loss()
            self.criterionSSIM  = SSIMLoss(data_range=2.0).to(self.device)

            # ---------- optimisers ----------
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(),
                                self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(),
                                self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # ------------------------------------------------------------------
    def set_input(self, input):
        """Unpack data from the dataloader."""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # ------------------------------------------------------------------
    def forward(self):
        """Run forward pass (shared by train and test)."""
        self.fake_B = self.netG_A(self.real_A)   # G_A(A)
        self.rec_A  = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)   # G_B(B)
        self.rec_B  = self.netG_A(self.fake_A)   # G_A(G_B(B))

    # ------------------------------------------------------------------
    def backward_D_basic(self, netD, real, fake):
        """Compute discriminator loss and call backward."""
        pred_real = netD(real)
        loss_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # ------------------------------------------------------------------
    def backward_G(self):
        """Compute all generator losses and call backward."""
        lambda_idt  = self.opt.lambda_identity
        lambda_A    = self.opt.lambda_A
        lambda_B    = self.opt.lambda_B
        lambda_ssim = self.opt.lambda_ssim

        # ---- Identity loss ----
        # (only when input_nc == output_nc; disabled by default for 2.5D)
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = torch.tensor(0.0, device=self.device)
            self.loss_idt_B = torch.tensor(0.0, device=self.device)

        # ---- GAN loss ----
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # ---- Cycle-consistency loss ----
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # ---- SSIM loss on reconstructions ----
        # rec_A should look like real_A; rec_B should look like real_B.
        # SSIM expects same number of channels — both pairs have identical nc.
        if lambda_ssim > 0:
            self.loss_ssim_A = self.criterionSSIM(self.rec_A, self.real_A) * lambda_ssim
            self.loss_ssim_B = self.criterionSSIM(self.rec_B, self.real_B) * lambda_ssim
        else:
            self.loss_ssim_A = torch.tensor(0.0, device=self.device)
            self.loss_ssim_B = torch.tensor(0.0, device=self.device)

        # ---- Combined generator loss ----
        self.loss_G = (
            self.loss_G_A + self.loss_G_B
            + self.loss_cycle_A + self.loss_cycle_B
            + self.loss_idt_A   + self.loss_idt_B
            + self.loss_ssim_A  + self.loss_ssim_B
        )
        self.loss_G.backward()

    # ------------------------------------------------------------------
    def optimize_parameters(self):
        """Single optimisation step called every training iteration."""
        self.forward()

        # Update Generators
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Update Discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

