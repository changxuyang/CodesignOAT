import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Src.Unet3DSim_HemisphericalArray import UNet

def run_inference(num_elements: int):
    # Parameter validation
    assert num_elements in (8, 16, 32, 64), "elements must be one of: 8, 16, 32 or 64"

    # Path configuration
    h5_path   = '../Data/SimTestData_3DHemisphericalArray.h5'
    ckpt_path = f'../Models/Checkpoint_Sim3D_HemisphericalArray_{num_elements}element.pkl'
    out_dir   = f'../Outputs/Results_Sim3DHemispherical_{num_elements}_elements'
    os.makedirs(out_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load raw data from HDF5 ---
    raw_ds = 'Inputs'
    with h5py.File(h5_path, 'r') as f:
        raw_np = f[raw_ds][:]
    raw_tensor = torch.from_numpy(raw_np).to(device)

    dataset = TensorDataset(raw_tensor)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Initialize model & load weights ---
    model = UNet(in_dim=1, out_dim=1, num_filters=4, num_sensors=num_elements).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # --- Inference & save combined images ---
    with torch.no_grad():
        for idx, (inp,) in enumerate(loader):
            inp_norm = (inp - inp.min()) / (inp.max() - inp.min())
            out, RecBP = model(inp_norm)

            out_np = out.squeeze().detach().cpu().numpy()
            RecBP_np = RecBP.squeeze().detach().cpu().numpy()

            # For visualization: MIP along depth
            CodesignMIP_xy = np.max(out_np, axis=2)
            BPMIP_xy = np.max(RecBP_np, axis=2)

            # Rescale to 0-255 for saving
            CodesignMIP_xy = 255 * (CodesignMIP_xy - CodesignMIP_xy.min()) / (CodesignMIP_xy.max() - CodesignMIP_xy.min())
            BPMIP_xy = 255 * (BPMIP_xy - BPMIP_xy.min()) / (BPMIP_xy.max() - BPMIP_xy.min())

            # Add text labels
            cv2.putText(BPMIP_xy, 'BP', (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            cv2.putText(CodesignMIP_xy, 'Codesign', (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)

            # Combine images side by side
            combined = np.concatenate([BPMIP_xy, CodesignMIP_xy], axis=1)
            save_path = os.path.join(out_dir, f'TestData_{idx:02d}.png')
            cv2.imwrite(save_path, combined)

    print(f'Inference completed. PNGs saved to: {out_dir}')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D Hemispherical Array inference script')
    parser.add_argument(
        '--elements', type=int, choices=[8, 16, 32, 64],
        default=64,
        help='Number of transducer elements to use (8, 16, 32, 64), default is 32'
    )
    args = parser.parse_args()
    run_inference(args.elements)