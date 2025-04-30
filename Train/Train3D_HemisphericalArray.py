import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm
from Src.Unet3DSim_HemisphericalArray import UNet
from Src.pytorch_ssim3D import SSIM3D
from PIL import Image

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

def muti_bce_loss_fusion(output, labels_v):
    loss_ssim =  1-ssim_loss(output, labels_v)
    loss_MSE = MSEloss(output,labels_v)
    loss = 100*(loss_MSE + 0.01*loss_ssim)
    return loss_ssim, loss_MSE, loss

def main():
    parser = argparse.ArgumentParser(description="Train U-Net")
    parser.add_argument('--train_h5',        type=str,   default='../Data/SimTrainData_3DHemisphericalArray.h5',
                        help='Path to the HDF5 file containing training data')
    parser.add_argument('--train_key',       type=str,   default='Inputs',
                        help='Key in HDF5 file for training dataset')
    parser.add_argument('--valid_h5',        type=str,   default='../Data/SimTestData_3DHemisphericalArray.h5',
                        help='Path to the HDF5 file containing validation data')
    parser.add_argument('--valid_key',       type=str,   default='Inputs',
                        help='Key in HDF5 file for validation dataset')
    parser.add_argument('--checkpoint_dir',  type=str,   default='../Train/TrainFiles/Checkpoint',
                        help='Directory to save model checkpoints')
    parser.add_argument('--stat_dir',        type=str,   default='../Train/TrainFiles/statistics',
                        help='Directory to save training statistics')
    parser.add_argument('--save_vis_dir',    type=str,   default='../Train/TrainFiles/CheckImage',
                        help='Directory to save visualization images')
    parser.add_argument('--batch_size',      type=int,   default=1,
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
    parser.add_argument('--TransducerNumber',type=int,   default=16,
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
    model = UNet(in_dim=1, out_dim=1, num_filters=4, num_sensors=args.TransducerNumber).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    mse_loss_fn = nn.MSELoss().to(device)
    ssim_fn = SSIM3D(window_size=11)

    # Load raw data
    with h5py.File(args.train_h5, 'r') as f:
        train_np = f[args.train_key][:].astype(np.float32)
    with h5py.File(args.valid_h5, 'r') as f:
        valid_np = f[args.valid_key][:].astype(np.float32)
    train_t = torch.from_numpy(train_np)
    valid_t = torch.from_numpy(valid_np)

    train_ds = TensorDataset(train_t)
    valid_ds = TensorDataset(valid_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)

    # Prepare statistics DataFrame
    stats = pd.DataFrame(columns=['Epoch', 'LR', 'TotalLoss', 'MSELoss', 'SSIMLoss'])

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

            mse_l = mse_loss_fn(outputs.squeeze(), targets.squeeze())
            ssim_l = 1.0 - ssim_fn(outputs, targets.unsqueeze(0))
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

                    v_outMIP_xy = v_out.squeeze().detach().cpu().numpy()
                    GTMIP_xy = vb_norm.squeeze().detach().cpu().numpy()
                    v_outMIP_xy = np.max(v_outMIP_xy.squeeze(), axis=2)
                    GTMIP_xy = np.max(GTMIP_xy.squeeze(), axis=2)

                    # Rescale [0,255]
                    v_outMIP_xy = 255 * (v_outMIP_xy - v_outMIP_xy.min()) / (
                            v_outMIP_xy.max() - v_outMIP_xy.min()
                    )
                    GTMIP_xy = 255 * (GTMIP_xy - GTMIP_xy.min()) / (
                            GTMIP_xy.max() - GTMIP_xy.min()
                    )

                    v_outMIP_xy = v_outMIP_xy.astype(np.uint8)
                    GTMIP_xy = GTMIP_xy.astype(np.uint8)

                    concat = np.concatenate([v_outMIP_xy, GTMIP_xy], axis=1)
                    img = Image.fromarray(concat)
                    img.save(
                        os.path.join(predict_vis_dir, f"epoch_{epoch_idx + 1}_sample_{vidx}.jpg")
                    )

    print("Training complete.")


if __name__ == "__main__":
    main()
