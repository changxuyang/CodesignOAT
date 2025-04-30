import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Src.Unet2DSim_CircularArray import Unet


def run_inference(num_elements: int):
    # Parameter validation
    assert num_elements in (4, 8, 16, 32), "elements must be one of: 4, 8, 16, or 32"

    # --- Path configuration ---
    h5_path   = '../Data/SimData_2DCircularArray.h5'
    ckpt_path = f'../Models/Checkpoint_Sim2D_CircularArray_{num_elements}element.pkl'
    out_dir   = f'../Outputs/Results_Sim2DCircular_{num_elements}_elements'
    os.makedirs(out_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load raw data and transducer positions from HDF5 ---
    rawtest_ds = 'test'
    with h5py.File(h5_path, 'r') as f:
        raw_np = f[rawtest_ds][:]    # shape: (80,1,128,128)
    raw_tensor = torch.from_numpy(raw_np).to(device)



    dataset = TensorDataset(raw_tensor)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Initialize model & load weights ---
    model = Unet(input_nc=1, num_sensors=num_elements).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
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
            inp = (inp - inp.min()) / (inp.max() - inp.min())
            output, RecBP = model(inp)
            out_crop = output.squeeze().cpu()[14:114, 14:114]
            RecBP_crop = RecBP.squeeze().cpu()[14:114, 14:114]

            codesign_img = normalize_to_uint8(out_crop)
            bp_img       = normalize_to_uint8(RecBP_crop)
            bp_img       = np.ascontiguousarray(bp_img)
            codesign_img = np.ascontiguousarray(codesign_img)


            cv2.putText(bp_img, 'BP', (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            cv2.putText(codesign_img, 'Codesign', (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)

            combined = np.concatenate([bp_img, codesign_img], axis=1)
            save_path = os.path.join(out_dir, f'TestData_{idx:02d}.png')
            cv2.imwrite(save_path, combined)

    print(f'Inference completed. Results saved to: {out_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Simulation Dataset Inference')
    parser.add_argument(
        '--elements', type=int, choices=[4, 8, 16, 32],
        default=4,
        help='Number of transducer elements to use (4, 8, 16, 32), default is 32'
    )
    args = parser.parse_args()
    run_inference(args.elements)
