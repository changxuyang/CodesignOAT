# 3D-UNet model.
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat


def trilinear_interpolate(images, x, y, z):

    N, H, W, D = images.shape
    images = images.permute(0, 3, 1, 2).unsqueeze(1)  # Reshape to (N, C, D, H, W)
    x = x / (W - 1) * 2.0 - 1.0
    y = y / (H - 1) * 2.0 - 1.0
    z = z / (D - 1) * 2.0 - 1.0

    xyz = torch.stack([x, y, z], dim=-1).repeat(N, 1, 1)
    xyz = xyz.unsqueeze(2)

    interpolated = F.grid_sample(images, xyz, mode='bilinear', padding_mode='zeros', align_corners=True)
    return interpolated.squeeze(3).squeeze(1)  # Reshape to (N, K, 1)

def forward_model_3DCUP(self, x):
    SensorPositionx, SensorPositiony, SensorPositionz = spherical_to_cartesian(self.r, self.SensorPosition_layer[:, 0],
                                                                               self.SensorPosition_layer[:, 1])

    output = torch.zeros(self.numberSamples, self.numberSensor, dtype=self.dtype).to(self.device)
    grid_diag = (torch.sqrt(
        torch.tensor(self.dx, dtype=self.dtype) ** 2 + torch.tensor(self.dy, dtype=self.dtype) ** 2 + torch.tensor(
            self.dz, dtype=self.dtype) ** 2)).to(self.device)

    delta_alpha = torch.pi / 4 / (50 - 1)
    delta_beta = torch.pi / 2 / (50 - 1)
    delta_d = (grid_diag / (50 - 1)).detach()

    x_grid = -(self.Nx / 2.0 - 0.5) * self.dx + self.dx * torch.arange(self.Nx).detach().to(self.device)
    y_grid = -(self.Ny / 2.0 - 0.5) * self.dy + self.dy * torch.arange(self.Ny).detach().to(self.device)
    z_grid = -(self.Nz / 2.0 - 0.5) * self.dz + self.dz * torch.arange(self.Nz).detach().to(self.device)

    X_pix, Y_pix, Z_pix = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    Dx = SensorPositionx[:, None, None, None] - X_pix
    Dy = SensorPositiony[:, None, None, None] - Y_pix
    Dz = SensorPositionz[:, None, None, None] - Z_pix

    Dist = torch.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2)
    Dist_xy = torch.sqrt(Dx ** 2 + Dy ** 2).detach()

    Alpha = torch.arctan2(torch.abs(Dy), torch.abs(Dx)).detach()
    Alpha[Alpha > torch.pi / 4] = torch.pi / 2 - Alpha[Alpha > torch.pi / 4].detach()
    Beta = torch.arctan2(torch.abs(Dz), Dist_xy).detach()

    Alpha_index = torch.clip(torch.ceil(Alpha / delta_alpha).long(), 0, 50 - 1).detach()
    Beta_index = torch.clip(torch.ceil(Beta / delta_beta).long(), 0, 50 - 1).detach()
    del Dx, Dy, Dz, Dist_xy, Alpha, Beta, X_pix, Y_pix, Z_pix

    batch_size = 2
    GradAcc = torch.zeros(200, dtype=torch.int)
    indices = torch.randperm(200)[:10]
    GradAcc[indices] = 1
    ii = 0

    for batch_start in range(0, self.numberSamples, batch_size):
        batch_end = min(batch_start + batch_size, self.numberSamples)
        batch_indices = torch.arange(batch_start, batch_end).to(self.device)
        D = (Dist[None, :] - self.sound_speed * self.t[batch_indices][:, None, None, None, None]).to(self.device)
        Factor = torch.where(D < 0, torch.tensor(-1.0).to(self.device), torch.tensor(1.0).to(self.device)).to(
            self.device)
        D = torch.abs(D)
        D_index = torch.clip(torch.ceil(D / delta_d).long(), 0, 50 - 1)
        dI = self.dI[D_index, Alpha_index, Beta_index]
        Element = -self.sound_speed / (Dist - D) * Factor * dI * x
        output[batch_start:batch_end, :] = torch.sum(Element, dim=(2, 3, 4))
        del D, Factor, D_index, dI, Element
        ii += 1

    return output

def Recon_BP_3DCUP(self, Rawdata):
    SensorPositionx, SensorPositiony, SensorPositionz = spherical_to_cartesian(self.r, self.SensorPosition_layer[:, 0],
                                                                               self.SensorPosition_layer[:, 1])

    bp_images = torch.zeros(1, self.Nx, self.Ny, self.Nz, dtype=self.dtype).to(self.device)

    # Generate the 3D grid
    x = torch.linspace(-0.5 * (self.Nx - 1) * self.dx, 0.5 * (self.Nx - 1) * self.dx, self.Nx).to(self.device) + \
        self.reconCenter[0]
    y = torch.linspace(-0.5 * (self.Ny - 1) * self.dy, 0.5 * (self.Ny - 1) * self.dy, self.Ny).to(self.device) + \
        self.reconCenter[1]
    z = torch.linspace(-0.5 * (self.Nz - 1) * self.dz, 0.5 * (self.Nz - 1) * self.dz, self.Nz).to(self.device) + \
        self.reconCenter[2]

    X, Y, Z = torch.meshgrid(x, y, z)

    for ii in range(self.numberSensor):
        T_Pt_batches = Rawdata[:, 0, :, ii]

        xd = X.to(self.device) - SensorPositionx[ii]
        yd = Y.to(self.device) - SensorPositiony[ii]
        zd = Z.to(self.device) - SensorPositionz[ii]
        d = torch.sqrt(xd ** 2 + yd ** 2 + zd ** 2)

        tbp = torch.round(
            (d * self.samplingFrequency) / self.sound_speed - self.t[0] * self.samplingFrequency + 1).long()
        tbp = torch.clamp(tbp, 0, self.numberSamples - 1)
        tbp_expanded = tbp.unsqueeze(0).expand(1, -1, -1, -1).flatten(start_dim=1)

        T_Pt_batches_expanded = T_Pt_batches.unsqueeze(-1).expand(-1, -1, self.Nx * self.Ny * self.Nz).to(self.device)

        T_Pt_expanded = torch.gather(T_Pt_batches_expanded, 1, tbp_expanded.unsqueeze(1)).view(1, self.Nx, self.Ny,
                                                                                               self.Nz)

        bp_images += T_Pt_expanded

    return bp_images

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates using PyTorch.

    Parameters:
    - r: Radius, a tensor.
    - theta: Polar angle (in radians), a tensor.
    - phi: Azimuthal angle (in radians), a tensor.

    Returns:
    - x, y, z: Cartesian coordinates, tensors.
    """
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates using PyTorch.

    Parameters:
    - x, y, z: Cartesian coordinates, tensors.

    Returns:
    - r: Radius, a tensor.
    - theta: Polar angle (in radians), a tensor.
    - phi: Azimuthal angle (in radians), a tensor.
    """
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    return r, theta, phi

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim), )

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters, num_sensors: int = 32,):
        super(UNet, self).__init__()
        self.device = 'cuda:0'
        self.dtype = torch.float32
        # Simulation space dimensions
        self.Nx, self.Ny, self.Nz = 64, 64, 32
        self.dx, self.dy, self.dz = 0.012 / (self.Nx - 1), 0.012 / (self.Ny - 1), 0.0032 / (self.Nz - 1)
        self.reconCenter = [0, 0, 0]
        self.sound_speed = 1500
        self.numberSensor = num_sensors
        self.numberSamples = 300  # 493
        self.samplingFrequency = 20e6
        nDelay = 420

        data = loadmat('../Data/Cup_Para.mat')
        trans_posi = data['transducerPos']
        trans_posi = torch.tensor(trans_posi[::int(512 / self.numberSensor), :], dtype=self.dtype).to(self.device)
        self.r, theta, phi = cartesian_to_spherical(trans_posi[:, 0], trans_posi[:, 1], trans_posi[:, 2])
        trans_posi_spherical = torch.stack((theta, phi), dim=1)
        self.dI = data['dI']
        self.dI = torch.tensor(self.dI, dtype=self.dtype).to(self.device)
        self.t = torch.arange(0, self.numberSamples) / self.samplingFrequency + nDelay / self.samplingFrequency
        self.t = torch.tensor(self.t, dtype=self.dtype).to(self.device)

        self.SensorPosition_layer = torch.nn.Parameter(trans_posi_spherical, requires_grad=True).to(device=self.device)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        Ax = forward_model_3DCUP(self, x.squeeze(0))
        RecBP = Recon_BP_3DCUP(self, Ax.unsqueeze(0).unsqueeze(0))


        # Down sampling
        down_1 = self.down_1(RecBP.unsqueeze(1))
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        # Bridge
        bridge = self.bridge(pool_5)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_5], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_4], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_3], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_2], dim=1)
        up_4 = self.up_4(concat_4)

        trans_5 = self.trans_5(up_4)
        concat_5 = torch.cat([trans_5, down_1], dim=1)
        up_5 = self.up_5(concat_5)

        # Output
        out = self.out(up_5)
        return out, RecBP
