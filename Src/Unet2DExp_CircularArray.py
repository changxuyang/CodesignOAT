import torch
import functools
import torch.nn as nn
import torch.nn.functional as F


def bilinear_interpolate(images, x, y):
    """
    Perform bilinear interpolation on a batch of images.

    Args:
        images (torch.Tensor): Input images of shape (N, 1, H, W).
        x (torch.Tensor): X coordinates of shape (1, K).
        y (torch.Tensor): Y coordinates of shape (1, K).

    Returns:
        torch.Tensor: Interpolated values of shape (N, K).
    """
    N, C, H, W = images.shape
    # Normalize coordinates to [-1, 1]
    x_norm = x / (W - 1) * 2 - 1
    y_norm = y / (H - 1) * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)
    grid = grid.repeat(N, 1, 1, 1)
    sampled = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled.squeeze(2).squeeze(1)


def BP_recon_2D(self, sigMat_batches, trans_posi_final, image_select):
    """
    Backprojection reconstruction

    Args:
        self: Model instance containing parameters.
        sigMat_batches (torch.Tensor): Signal batches of shape (B, 1, T, num_sensors).
        trans_posi_final (torch.Tensor): Transducer positions of shape (num_sensors, 2).
        image_select (int): Option index.

    Returns:
        torch.Tensor: Reconstructed images of shape (B, 1, n1, n2).
    """
    c0 = self.sound_speed
    resolution = [self.Nx - 2 * self.pml_size, self.Ny - 2 * self.pml_size]
    recon_dims = [0.025, 0.025]
    recon_center = [0, 0]
    # Time axis with original offset
    ts = torch.arange(0, self.Nt, dtype=torch.float64, device=self.device) / 40e6 + 113/40e6

    x_sensor = trans_posi_final[:, 0]
    y_sensor = trans_posi_final[:, 1]
    fs = 1 / (ts[1] - ts[0])
    sizeT = len(ts)
    n1, n2 = resolution
    B = sigMat_batches.size(0)

    bp_images = torch.zeros(B, 1, n1, n2, device=self.device)
    Dx = recon_dims[0] / (n1 - 1)
    Dy = recon_dims[1] / (n2 - 1)
    x = torch.linspace(-0.5*(n1-1)*Dx, 0.5*(n1-1)*Dx, n1, device=self.device) + recon_center[0]
    y = torch.linspace(-0.5*(n2-1)*Dy, 0.5*(n2-1)*Dy, n2, device=self.device) + recon_center[1]
    X, Y = torch.meshgrid(x, y)

    for i in range(self.num_sensors):
        signals = sigMat_batches[:, 0, :, i]  # shape: (B, T)

        d = torch.sqrt((X - x_sensor[i])**2 + (Y - y_sensor[i])**2)
        tbp = torch.round(d * fs / c0 - ts[0] * fs + 1).long()
        tbp = torch.clamp(tbp, 0, sizeT - 1)

        tbp_exp = tbp.unsqueeze(0).expand(B, -1, -1).flatten(start_dim=1)
        sig_exp = signals.unsqueeze(-1).expand(-1, -1, n1*n2)
        pts = torch.gather(sig_exp, 1, tbp_exp.unsqueeze(1)).view(B, n1, n2)
        bp_images[:, 0] += pts

    # Average over sensors
    bp_images /= self.num_sensors
    # Flip and rotate to correct orientation
    bp_images = torch.flip(bp_images, dims=[3])
    bp_images = torch.rot90(bp_images, k=1, dims=[2, 3])
    return bp_images


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines a U-Net submodule with skip connections.
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, final_activation=None):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        # Downsampling
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        # Upsampling
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            layers = [downconv, submodule, uprelu, upconv]
            if final_activation:
                layers.append(final_activation)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            layers = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            layers = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)
        self.outermost = outermost

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class Unet(nn.Module):
    """
    U-Net based generator for image reconstruction.

    Args:
        input_nc (int): Number of input channels.
        num_sensors (int): Number of sensors (4, 8, 16, 32).
        output_nc (int): Number of output channels.
        num_downs (int): Number of downsampling layers.
        ngf (int): Number of filters in the last conv layer.
        norm_layer: Normalization layer.
        use_dropout (bool): Whether to use dropout.
        final_activation: Activation at the final layer.
    """
    def __init__(self, input_nc, num_sensors=32, output_nc=1,
                 num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, final_activation=nn.Sigmoid()):
        super().__init__()
        # Build the U-Net structure
        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf*2, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc, unet_block, outermost=True, final_activation=final_activation)

        # Simulation parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64
        self.num_sensors = num_sensors
        self.Nx = 256 + 2*3  # with PML padding
        self.Ny = 256 + 2*3
        self.pml_size = 3
        NN = 0.025
        self.dx = NN / (self.Nx - 2*self.pml_size)
        self.dy = NN / (self.Ny - 2*self.pml_size)
        self.Nt = 2029
        self.sound_speed = 1572
        self.ts = torch.arange(0, self.Nt, dtype=torch.float64, device=self.device) / 40e6

    def forward(self, input, trans_positions):
        # Backprojection reconstruction
        RecBP = BP_recon_2D(self, input, trans_positions, image_select=1)
        # Normalize RecBP
        RecBP = (RecBP - RecBP.min()) / (RecBP.max() - RecBP.min())
        # U-Net generation
        out = self.model(RecBP)
        return out, RecBP
