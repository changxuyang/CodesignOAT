import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Src.Unet2DExp_CircularArray import Unet

def run_inference(num_elements: int):
    # Parameter validation
    assert num_elements in (4, 8, 16, 32), "elements must be one of: 4, 8, 16, or 32"

    # Path configuration
    h5_path   = '../Data/ExpTestData_2DCircularArray.h5'
    ckpt_path = f'../Models/Checkpoint_Exp2D_CircularArray_{num_elements}element.pkl'
    out_dir   = f'../Outputs/Results_Exp2DCircular_{num_elements}_elements'
    os.makedirs(out_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load raw data and transducer positions from HDF5 ---
    raw_ds = f'Rawdata_{num_elements}elements'
    pos_ds = f'TransPositions_{num_elements}elements'
    with h5py.File(h5_path, 'r') as f:
        raw_np = f[raw_ds][:]    # shape: (50,1,2029,N)
        pos_np = f[pos_ds][:]    # shape: (N,2)
    raw_tensor = torch.from_numpy(raw_np).to(device)
    pos_tensor = torch.from_numpy(pos_np).to(device)

    dataset = TensorDataset(raw_tensor)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Initialize model & load weights ---
    model = Unet(input_nc=1, num_sensors=num_elements).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Helper: normalize tensor to uint8 image
    def normalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.squeeze().cpu().numpy().astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return (arr * 255).astype(np.uint8)

    # --- Inference & save combined images ---
    with torch.no_grad():
        for idx, (inp,) in enumerate(loader):
            # Forward pass
            out, RecBP = model(inp, pos_tensor)
            codesign_img = normalize_to_uint8(out)
            bp_img       = normalize_to_uint8(RecBP)
            bp_img       = np.ascontiguousarray(bp_img)
            codesign_img = np.ascontiguousarray(codesign_img)
            cv2.putText(bp_img,       'BP',       (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            cv2.putText(codesign_img, 'Codesign',(5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)

            combined = np.concatenate([bp_img, codesign_img], axis=1)
            save_path = os.path.join(out_dir, f'TestData_{idx:02d}.png')
            cv2.imwrite(save_path, combined)

    print(f'Inference completed. Results saved to: {out_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='2D Circular Array inference script')
    parser.add_argument(
        '--elements', type=int, choices=[4, 8, 16, 32],
        default=4,
        help='Number of transducer elements to use (4, 8, 16, 32), default is 32'
    )
    args = parser.parse_args()
    run_inference(args.elements)
