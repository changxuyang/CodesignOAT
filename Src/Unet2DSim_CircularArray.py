import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple

def cartesian_to_polar(
    cartesian_coords: torch.Tensor,
    grid_size: int = 128
) -> torch.Tensor:
    """
    Convert Cartesian coordinates to polar coordinates (r, θ).

    Parameters:
        cartesian_coords (torch.Tensor): (N x 2) Cartesian coordinates.
        grid_size (int): Size of the square grid.

    Returns:
        torch.Tensor: (N x 2) [r, θ].
    """
    cx = cy = grid_size / 2
    x = cartesian_coords[:, 0] - cx
    y = cartesian_coords[:, 1] - cy
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack((r, theta), dim=-1)

def polar_to_cartesian(
    polar_coords: torch.Tensor,
    grid_size: int = 128
) -> torch.Tensor:
    """
    Convert polar coordinates (r, θ) back to Cartesian coordinates.

    Parameters:
        polar_coords (torch.Tensor): (N x 2) [r, θ].
        grid_size (int): Size of the square grid.

    Returns:
        torch.Tensor: (N x 2) Cartesian coordinates.
    """
    cx = cy = grid_size / 2
    r = polar_coords[:, 0]
    theta = polar_coords[:, 1]
    x = r * torch.cos(theta) + cx
    y = r * torch.sin(theta) + cy
    return torch.stack((x, y), dim=-1)

def bilinear_interpolate(
    self,
    images: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Perform bilinear interpolation on a batch of images at the same set of points.

    Parameters:
        images (torch.Tensor): (N, 1, H, W) image tensor.
        x (torch.Tensor): (1, K) X coordinates.
        y (torch.Tensor): (1, K) Y coordinates.

    Returns:
        torch.Tensor: (N, K, 1) interpolated values.
    """
    N, C, H, W = images.shape
    x_norm = x / (W - 1) * 2.0 - 1.0
    y_norm = y / (H - 1) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1) \
               .repeat(N, 1, 1) \
               .unsqueeze(2) \
               .to(self.dtype)
    sampled = F.grid_sample(
        images,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return sampled.squeeze(3)

def create_circular_sensor(
    num_sensors: int,
    radius: float,
    grid_size: int
) -> np.ndarray:
    """
    Generate a circular transducer array.

    Parameters:
        num_sensors (int): Number of element.
        radius (float): Radius of the circle.
        grid_size (int): Size of the grid.

    Returns:
        np.ndarray: (grid_size, grid_size) mask matrix with element locations set to 1.
    """
    angles = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)
    x = (radius * np.cos(angles) + grid_size // 2 - 1).astype(int)
    y = (radius * np.sin(angles) + grid_size // 2 - 1).astype(int)
    x = np.clip(x, 0, grid_size - 1)
    y = np.clip(y, 0, grid_size - 1)
    mask = np.zeros((grid_size, grid_size), dtype=int)
    mask[x, y] = 1
    return mask

class Unet(nn.Module):
    """
    U-Net model integrating forward wave simulation and BP reconstruction.
    """
    def __init__(
        self,
        input_nc: int,
        output_nc: int = 1,
        num_downs: int = 5,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        num_sensors: int = 32,
    ):
        super(Unet, self).__init__()
        self.cuda = 'cuda:0'
        self.dtype = torch.float64

        # Build U-Net skip connection structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout
            )
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf,      ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            final_activation=nn.Sigmoid()
        )

        # Simulation and grid parameters
        self.Nx, self.Ny, self.Nt = 128, 128, 512
        NN = 1.5e-2
        self.dx, self.dy = NN / self.Nx, NN / self.Ny
        self.dt = 1 / 20e6
        self.sound_speed = 1500.0
        self.numberSensor = num_sensors
        self.pml_size = 3
        self.Nx += 2 * self.pml_size
        self.Ny += 2 * self.pml_size
        self.ts = torch.arange(0, self.Nt) / 20e6

        # k-space operators
        kx = torch.arange(-self.Nx // 2, self.Nx // 2, dtype=self.dtype)
        kx = (2 * torch.pi / self.dx) * (kx / self.Nx).unsqueeze(1)
        ky = kx.t()
        k  = torch.sqrt(kx**2 + ky**2)
        self.ddx_k_shift_pos = torch.fft.ifftshift(1j * kx * torch.exp(1j * kx * self.dx / 2)).to(device=self.cuda)
        self.ddx_k_shift_neg = torch.fft.ifftshift(1j * kx * torch.exp(-1j * kx * self.dx / 2)).to(device=self.cuda)
        self.ddy_k_shift_pos = torch.fft.ifftshift(1j * ky * torch.exp(1j * ky * self.dy / 2)).to(device=self.cuda)
        self.ddy_k_shift_neg = torch.fft.ifftshift(1j * ky * torch.exp(-1j * ky * self.dy / 2)).to(device=self.cuda)
        self.kappa          = torch.fft.ifftshift(torch.sinc(self.sound_speed * k * self.dt / (2 * torch.pi))).to(device=self.cuda)

        # PML absorption coefficients
        pml_alpha = 2
        idx = torch.arange(1, self.pml_size + 1, dtype=self.dtype)
        left1  = pml_alpha * (self.sound_speed / self.dx) * ((idx - self.pml_size - 1) / (-self.pml_size))**4
        right1 = pml_alpha * (self.sound_speed / self.dx) * (idx / self.pml_size)**4
        left2  = pml_alpha * (self.sound_speed / self.dx) * (((idx - self.pml_size - 1) + 0.5) / (-self.pml_size))**4
        right2 = pml_alpha * (self.sound_speed / self.dx) * ((idx + 0.5) / self.pml_size)**4
        pml1 = torch.exp(-torch.cat([left1,
                                     torch.ones(self.Nx - 2 * self.pml_size),
                                     right1]) * self.dt / 2)
        pml2 = torch.exp(-torch.cat([left2,
                                     torch.ones(self.Nx - 2 * self.pml_size),
                                     right2]) * self.dt / 2)
        self.pml_x    = pml1.unsqueeze(1).to(device=self.cuda)
        self.pml_y    = pml1.unsqueeze(0).to(device=self.cuda)
        self.pml_sg_x = pml2.unsqueeze(1).to(device=self.cuda)
        self.pml_sg_y = pml2.unsqueeze(0).to(device=self.cuda)

        # Sensor geometry
        mask = create_circular_sensor(self.numberSensor, 60, self.Nx - self.pml_size)
        mask = torch.from_numpy(mask).to(device=self.cuda)
        SensorPosition = torch.stack([
            mask.nonzero(as_tuple=True)[0].float(),
            mask.nonzero(as_tuple=True)[1].float()
        ], dim=1)
        polar = cartesian_to_polar(SensorPosition, grid_size=self.Nx - self.pml_size)
        self.SensorPosition_polar_r = 60 * torch.ones_like(polar[:,1]).to(device=self.cuda)
        self.SensorPosition_layer   = nn.Parameter(polar[:,1], requires_grad=True).to(device=self.cuda)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pad input in the PML region
        pad = self.pml_size
        b, c, _, _ = input.shape
        sim_in = torch.zeros((b, c, self.Nx, self.Ny), device=self.cuda, dtype=input.dtype)
        sim_in[:, :, pad:-pad, pad:-pad] = input
        y = self.forward_model(sim_in)
        RecBP = self.BP_parallel(y)
        return self.model(RecBP), RecBP

    def forward_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward wave simulation and sample at transducer element positions.
        Returns (batch, 1, numberSensor, Nt).
        """
        # Initialize fields
        b = x.size(0)
        Nx, Ny, Nt = self.Nx, self.Ny, self.Nt
        rhox = rhoy = ux_sgx = uy_sgy = p_k = torch.zeros((b, 1, Nx, Ny),
                                                         dtype=torch.float64,
                                                         device=self.cuda)
        # element positions
        Position = torch.stack((self.SensorPosition_polar_r, self.SensorPosition_layer), dim=1)
        SensorPosition = polar_to_cartesian(Position).to(device=self.cuda)
        SensorPosition += self.pml_size

        sensor_data = torch.zeros((b, self.numberSensor, Nt),
                                  dtype=self.dtype,
                                  device=self.cuda)
        for t in range(Nt):
            ux_sgx = self.pml_sg_x * ((self.pml_sg_x * ux_sgx)
                      - self.dt * torch.fft.ifftn(self.ddx_k_shift_pos * (self.kappa * p_k)).real)
            uy_sgy = self.pml_sg_y * ((self.pml_sg_y * uy_sgy)
                      - self.dt * torch.fft.ifftn(self.ddy_k_shift_pos * (self.kappa * p_k)).real)

            duxdx = torch.fft.ifftn(self.ddx_k_shift_neg * (self.kappa * torch.fft.fftn(ux_sgx))).real
            duydy = torch.fft.ifftn(self.ddy_k_shift_neg * (self.kappa * torch.fft.fftn(uy_sgy))).real

            rhox = self.pml_x * ((self.pml_x * rhox) - self.dt * duxdx)
            rhoy = self.pml_y * ((self.pml_y * rhoy) - self.dt * duydy)

            p = self.sound_speed**2 * (rhox + rhoy)

            if t == 0:
                p.copy_(x)
                rhox.copy_(x / (2 * self.sound_speed**2))
                rhoy.copy_(x / (2 * self.sound_speed**2))
                ux_sgx.copy_( self.dt * torch.fft.ifftn(self.ddx_k_shift_pos * (self.kappa * torch.fft.fftn(p))).real / 2 )
                uy_sgy.copy_( self.dt * torch.fft.ifftn(self.ddy_k_shift_pos * (self.kappa * torch.fft.fftn(p))).real / 2 )

            p_k = torch.fft.fftn(p)

            p_int = bilinear_interpolate(self,
                                         p,
                                         SensorPosition[:,0],
                                         SensorPosition[:,1])
            sensor_data[:, :, t] = p_int[:, 0, :]

        return sensor_data.unsqueeze(1)

    def BP_parallel(self, rawdata):
        """
        returns (batch, 1, Nx-2*pml, Ny-2*pml),
        """
        # Restore original element position calculation logic
        Position = torch.stack((self.SensorPosition_polar_r, self.SensorPosition_layer), dim=1)
        SensorPosition = polar_to_cartesian(Position)

        # Prepare raw data and output tensor
        raw_data = rawdata.to(device=self.cuda)
        n_batches = raw_data.size(0)
        n1 = self.Nx - 2 * self.pml_size
        n2 = self.Ny - 2 * self.pml_size
        reconstructed_images = torch.zeros(
            n_batches, 1, n1, n2,
            device=self.cuda
        )

        # Generate pixel grid
        x, y = torch.meshgrid(
            torch.arange(n1, device=self.cuda),
            torch.arange(n2, device=self.cuda),
        )
        pixel_positions = torch.stack([x, y], dim=-1) * self.dx

        # Compute distances, time delays, and indices
        distances = torch.norm(
            pixel_positions.unsqueeze(2) - (SensorPosition * self.dx),
            dim=3
        )
        time_delays = distances / self.sound_speed
        time_indices = (time_delays / self.dt).long()

        # Main loop: accumulate for each batch and sensor
        for b in range(n_batches):
            # Whether each pixel has a valid index
            valid_indices = time_indices < self.Nt
            # Initialize accumulated signal
            sum_signals = torch.zeros(n1, n2, device=self.cuda)

            for si in range(SensorPosition.shape[0]):
                ti = time_indices[:, :, si]
                mask = valid_indices[:, :, si]
                profile = raw_data[b, 0, si, :]  # (Nt,)
                sensor_img = torch.zeros_like(sum_signals)  # (n1, n2)

                flat_idx = ti[mask]  # valid flat indices
                vals = profile[flat_idx].float()
                sensor_img[mask] = vals

                sum_signals += sensor_img

            # Average and write to output
            reconstructed_images[b, 0] = (
                    sum_signals / SensorPosition.shape[0]
            ).to(device=self.cuda)

        # Flip and rotate to match the original version exactly
        reconstructed_images = torch.flip(reconstructed_images, dims=[3])
        reconstructed_images = torch.rot90(
            reconstructed_images, k=1, dims=[2, 3]
        )
        return reconstructed_images


class UnetSkipConnectionBlock(nn.Module):
    """
    U-Net submodule with skip connections.
    """
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int = None,
        submodule: nn.Module = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        final_activation=None
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        if input_nc is None:
            input_nc = outer_nc

        # down
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        # up
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            if final_activation is not None:
                up.append(final_activation)
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
