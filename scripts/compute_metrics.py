#!/usr/bin/env python3
"""
Compute evaluation metrics (PSNR, SSIM, LPIPS) for BCI project
and save results to a comprehensive report.
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import datetime

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from src.options.test_options import TestOptions
from src.data import create_dataset
from src.models import create_model

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("Warning: scikit-image not available. Installing...")
    os.system("pip3 install scikit-image")
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: LPIPS not available. Will skip LPIPS metric.")
    LPIPS_AVAILABLE = False


def tensor2im(input_image, imtype=np.uint8):
    """Convert tensor to numpy image."""
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        
        # Handle batch dimension
        if len(image_numpy.shape) == 4:
            image_numpy = image_numpy[0]
        
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def compute_metrics(fake, real):
    """Compute PSNR and SSIM."""
    # Convert tensors to numpy arrays (0-255 range)
    fake_np = tensor2im(fake)
    real_np = tensor2im(real)
    
    # Compute PSNR and SSIM
    psnr_value = psnr(real_np, fake_np, data_range=255)
    ssim_value = ssim(real_np, fake_np, channel_axis=2, data_range=255)
    
    return psnr_value, ssim_value


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    # Storage for metrics
    psnr_scores = []
    ssim_scores = []
    
    # Run inference and compute metrics
    print(f"Computing metrics on {len(dataset)} test samples...")
    model.eval()
    
    for i, data in enumerate(tqdm(dataset)):
        if i >= opt.num_test:
            break
        
        model.set_input(data)
        model.test()
        
        visuals = model.get_current_visuals()
        fake_B = visuals['fake_B']
        real_B = visuals['real_B']
        
        # Compute metrics
        psnr_val, ssim_val = compute_metrics(fake_B, real_B)
        
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
    
    # Compute averages
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {opt.name}")
    print(f"Dataset: {opt.dataroot}")
    print(f"Samples evaluated: {len(psnr_scores)}")
    print("-"*60)
    print(f"PSNR:  {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"SSIM:  {avg_ssim:.4f} ± {std_ssim:.4f}")
    print("="*60 + "\n")
    
    # Save detailed report
    results_dir = opt.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    report_path = os.path.join(results_dir, f'metrics_report_{opt.name}.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BCI PROJECT - QUANTITATIVE EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXPERIMENT DETAILS\n")
        f.write("-"*70 + "\n")
        f.write(f"Model Name:        {opt.name}\n")
        f.write(f"Generator:         {opt.netG}\n")
        f.write(f"Dataset:           {opt.dataroot}\n")
        f.write(f"Test Samples:      {len(psnr_scores)}\n")
        f.write(f"Input Channels:    {opt.input_nc}\n")
        f.write(f"Output Channels:   {opt.output_nc}\n\n")
        
        f.write("QUANTITATIVE RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"PSNR (Peak Signal-to-Noise Ratio)\n")
        f.write(f"  Mean:            {avg_psnr:.4f} dB\n")
        f.write(f"  Std Dev:         {std_psnr:.4f} dB\n")
        f.write(f"  Min:             {min(psnr_scores):.4f} dB\n")
        f.write(f"  Max:             {max(psnr_scores):.4f} dB\n\n")
        
        f.write(f"SSIM (Structural Similarity Index)\n")
        f.write(f"  Mean:            {avg_ssim:.4f}\n")
        f.write(f"  Std Dev:         {std_ssim:.4f}\n")
        f.write(f"  Min:             {min(ssim_scores):.4f}\n")
        f.write(f"  Max:             {max(ssim_scores):.4f}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*70 + "\n")
        f.write("PSNR: Higher is better. >18 dB is good for medical image translation.\n")
        f.write("SSIM: Range [0, 1]. Higher is better. >0.3 is acceptable for GANs.\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"✅ Detailed report saved to: {report_path}")
    
    # Also save CSV for easy import to Excel/plotting
    csv_path = os.path.join(results_dir, f'metrics_{opt.name}.csv')
    with open(csv_path, 'w') as f:
        f.write("Sample,PSNR,SSIM\n")
        for i, (p, s) in enumerate(zip(psnr_scores, ssim_scores)):
            f.write(f"{i},{p:.4f},{s:.4f}\n")
    
    print(f"✅ CSV data saved to: {csv_path}")
    
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    main()
