import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Src.Unet3DExp_HemisphericalArray import UNet
import scipy.io as sio

def run_inference():
    # Path configuration
    h5_path   = '../Data/ExpTestData_3DHemisphericalArray_RtHand_thmFinger_FT.h5'
    ckpt_path = '../Models/Checkpoint_Exp3D_HemisphericalArray_32element.pkl'
    out_dir   = '../Outputs/Results_Exp3DHemispherical_32elements_RtHand_thmFinger_FT'
    os.makedirs(out_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load raw data from HDF5 ---
    raw_ds = 'BPRec_Codesign'
    with h5py.File(h5_path, 'r') as f:
        raw_np = f[raw_ds][:]
    raw_tensor = torch.from_numpy(raw_np).to(device)

    dataset = TensorDataset(raw_tensor)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Initialize model & load weights ---
    model = UNet(in_dim=1, out_dim=1, num_filters=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Prepare container for all outputs
    outputs = []

    # --- Inference & save combined images ---
    with torch.no_grad():
        for idx, (inp,) in enumerate(loader):
            inp_norm = (inp - inp.min()) / (inp.max() - inp.min())
            out = model(inp_norm)

            out_np = out.squeeze().detach().cpu().numpy()
            outputs.append(out_np)

            # For visualization: MIP along depth
            CodesignMIP_xy = np.max(out_np, axis=2)
            BPMIP_xy = np.max(inp_norm.squeeze().detach().cpu().numpy(), axis=2)

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

    # Stack all outputs
    all_outputs = np.stack(outputs, axis=-1)

    # Save to a .mat file
    mat_path = os.path.join(out_dir, 'Matresults.mat')
    sio.savemat(mat_path, {'results': all_outputs})

    print(f'Inference completed. PNGs saved to: {out_dir}')
    print(f'MAT file saved to: {mat_path}')


if __name__ == '__main__':
    run_inference()