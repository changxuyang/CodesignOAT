import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm

from Src.Unet2DSim_CircularArray import Unet
from Src.pytorch_ssim import SSIM


def plot_and_save_cartesian_coords(coords: torch.Tensor, filename: str) -> None:
    """
    Plot element positions in Cartesian coordinates and save to a JPG file.
    Does not display interactively; closes the figure to free memory.

    Args:
        coords: Tensor of shape (N, 2) containing x,y positions.
        filename: Path where the JPG will be saved.
    """
    xy = coords.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy[:, 0], xy[:, 1], c='blue', marker='o')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Element Positions')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    fig.savefig(filename, format='jpg', bbox_inches='tight')
    plt.close(fig)
def polar_to_cartesian(polar_coords: torch.Tensor, grid_size: int = 128) -> torch.Tensor:
    """
    Convert polar coordinates (r, theta) to Cartesian (x, y) on a square grid.

    Args:
        polar_coords: Tensor of shape (N, 2), columns are (r, theta).
        grid_size: Side length of the square grid.

    Returns:
        Tensor of shape (N, 2) with Cartesian coordinates.
    """
    center = grid_size / 2
    r = polar_coords[:, 0]
    theta = polar_coords[:, 1]
    x = r * torch.cos(theta) + center
    y = r * torch.sin(theta) + center
    return torch.stack((x, y), dim=-1)

def main():
    parser = argparse.ArgumentParser(description="Train U-Net with per-batch min-max normalization")
    parser.add_argument('--train_h5',        type=str,   default='../Data/SimData_2DCircularArray.h5',
                        help='Path to the HDF5 file containing training and validation data')
    parser.add_argument('--train_key',       type=str,   default='train',
                        help='Key in HDF5 file for training dataset')
    parser.add_argument('--valid_key',       type=str,   default='test',
                        help='Key in HDF5 file for validation dataset')
    parser.add_argument('--checkpoint_dir',  type=str,   default='../Train/TrainFiles/Checkpoint',
                        help='Directory to save model checkpoints')
    parser.add_argument('--stat_dir',        type=str,   default='../Train/TrainFiles/statistics',
                        help='Directory to save training statistics')
    parser.add_argument('--save_vis_dir',    type=str,   default='../Train/TrainFiles/CheckImage',
                        help='Directory to save visualization images')
    parser.add_argument('--batch_size',      type=int,   default=8,
                        help='Training batch size')
    parser.add_argument('--start_epoch',     type=int,   default=0,
                        help='Starting epoch number')
    parser.add_argument('--epochs',          type=int,   default=100,
                        help='Total number of epochs to train')
    parser.add_argument('--epochs_ckpt',     type=int,   default=10,
                        help='Save a checkpoint every N epochs')
    parser.add_argument('--lr',              type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--suffix',          type=str,   default='20250501',
                        help='Suffix for output files')
    parser.add_argument('--device',          type=str,   default='cuda:0',
                        help='PyTorch device (e.g. "cuda:0" or "cpu")')
    parser.add_argument('--TransducerNumber',type=int,   default=32,
                        help='Number of elements in the circular array')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.stat_dir, exist_ok=True)
    sensor_vis_dir  = os.path.join(args.save_vis_dir, 'ElementPosition')
    predict_vis_dir = os.path.join(args.save_vis_dir, 'Predict')
    os.makedirs(sensor_vis_dir,  exist_ok=True)
    os.makedirs(predict_vis_dir, exist_ok=True)
    device = torch.device(args.device)

    # Instantiate model, optimizer, scheduler, and loss functions
    model = Unet(input_nc=1, num_sensors=args.TransducerNumber).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    mse_loss_fn = nn.MSELoss().to(device)
    ssim_fn = SSIM()

    # Load raw data
    with h5py.File(args.train_h5, 'r') as f:
        train_np = f[args.train_key][:].astype(np.float32)  # shape: (N,1,H,W)
        valid_np = f[args.valid_key][:].astype(np.float32)

    train_t = torch.from_numpy(train_np)
    valid_t = torch.from_numpy(valid_np)

    train_ds = TensorDataset(train_t)
    valid_ds = TensorDataset(valid_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)

    # Prepare statistics DataFrame
    stats = pd.DataFrame(columns=['Epoch', 'LR', 'TotalLoss', 'MSELoss', 'SSIMLoss'])
    crop_start, crop_end = 14, 114

    # Training loop
    for epoch in range(args.epochs):
        epoch_idx = epoch + args.start_epoch
        model.train()

        running_total = running_mse = running_ssim = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch_idx+1}/{args.start_epoch+args.epochs}",
                    unit='batch', ncols=100)

        for batch_idx, (batch_tensor,) in enumerate(pbar):
            batch = batch_tensor.to(device)  # shape: (B,1,H,W)

            # Per-batch min–max normalization to [0,1]
            bmin = batch.view(batch.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
            bmax = batch.view(batch.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
            batch_norm = (batch - bmin) / (bmax - bmin)

            inputs, targets = batch_norm, batch_norm

            optimizer.zero_grad()
            outputs, BPRec = model(inputs)

            # Crop edges before computing loss
            out_crop = outputs[..., crop_start:crop_end, crop_start:crop_end]
            tgt_crop = targets[..., crop_start:crop_end, crop_start:crop_end]

            mse_l = mse_loss_fn(out_crop, tgt_crop)
            ssim_l = 1.0 - ssim_fn(out_crop, tgt_crop)
            loss = 100 * (mse_l + 0.01 * ssim_l)

            loss.backward()
            optimizer.step()

            running_total += loss.item()
            running_mse   += mse_l.item()
            running_ssim  += ssim_l.item()
            pbar.set_postfix({
                'Total': f"{running_total/(batch_idx+1):.4f}",
                'MSE':   f"{running_mse/(batch_idx+1):.4f}",
                'SSIM':  f"{running_ssim/(batch_idx+1):.4f}"
            })

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        avg_total = running_total / len(train_loader)
        avg_mse   = running_mse   / len(train_loader)
        avg_ssim  = running_ssim  / len(train_loader)

        print(f"[Epoch {epoch_idx+1}] LR={lr:.1e}  "
              f"TotalLoss={avg_total:.4f}  MSELoss={avg_mse:.4f}  SSIMLoss={avg_ssim:.4f}")

        # Log statistics
        stats.loc[len(stats)] = {
            'Epoch':      epoch_idx+1,
            'LR':         lr,
            'TotalLoss':  avg_total,
            'MSELoss':    avg_mse,
            'SSIMLoss':   avg_ssim
        }
        stats.to_excel(os.path.join(args.stat_dir, f"training_loss-{args.suffix}.xlsx"), index=False)

        # Save element positions plot
        with torch.no_grad():
            pos = torch.stack((model.SensorPosition_polar_r, model.SensorPosition_layer), dim=1)
            sensor_xy = polar_to_cartesian(pos, grid_size=model.Nx - model.pml_size)
            plot_and_save_cartesian_coords(
                sensor_xy,
                os.path.join(sensor_vis_dir, f"epoch_{epoch_idx+1}.jpg")
            )

        # Checkpoint and validation visualization
        if (epoch_idx + 1) % args.epochs_ckpt == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch_idx+1}_{args.suffix}.pth")
            torch.save(model.state_dict(), ckpt_path)

            model.eval()
            with torch.no_grad():
                for vidx, (v_batch,) in enumerate(valid_loader):
                    vb = v_batch.to(device)
                    vmin = vb.view(-1).min(); vmax = vb.view(-1).max()
                    vb_norm = (vb - vmin) / (vmax - vmin)
                    v_out, v_BPRec = model(vb_norm)

                    out_crop = v_out[..., crop_start:crop_end, crop_start:crop_end]
                    gt_crop  = vb_norm[..., crop_start:crop_end, crop_start:crop_end]

                    save_image(
                        torch.cat([out_crop[0], gt_crop[0]], dim=1),
                        os.path.join(predict_vis_dir, f"epoch_{epoch_idx+1}_sample_{vidx}.jpg")
                    )

    print("Training complete.")


if __name__ == "__main__":
    main()
