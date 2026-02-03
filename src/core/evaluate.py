import os
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from src.metrics.lpips_metric import LPIPSMetric

def parse_opt():
    #Set train options
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--result_path', type=str, default='./results/pyramidpix2pix', help='results saved path')
    parser.add_argument('--gpu', action='store_true', help='use gpu for lpips')
    opt = parser.parse_args()
    return opt

def to_tensor(img_cv):
    # IMG is BGR from cv2
    img_rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb) # [0, 1] C,H,W
    img_tensor = (img_tensor * 2.0) - 1.0 # [-1, 1]
    return img_tensor.unsqueeze(0) # 1, C, H, W

def evaluate_metrics(result_path, use_gpu=True):
    psnr = []
    ssim = []
    lpips_scores = []
    
    try:
        lpips_cal = LPIPSMetric(use_gpu=use_gpu)
    except Exception as e:
        print(f"Warning: Could not initialize LPIPS: {e}")
        lpips_cal = None
    
    image_dir = os.path.join(result_path, 'test_latest/images')
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} not found.")
        return

    print(f"Evaluating images in {image_dir}")
    
    for i in tqdm(os.listdir(image_dir)):
        if 'fake_B' in i:
            try:
                fake_path = os.path.join(image_dir, i)
                real_path = os.path.join(image_dir, i.replace('fake_B', 'real_B'))
                
                if not os.path.exists(real_path):
                    continue

                fake = cv.imread(fake_path)
                real = cv.imread(real_path)
                
                if fake is None or real is None:
                    print(f"Failed to read images: {fake_path} or {real_path}")
                    continue

                # PSNR & SSIM (using numpy/skimage)
                PSNR = peak_signal_noise_ratio(fake, real)
                psnr.append(PSNR)
                SSIM = structural_similarity(fake, real, multichannel=True)
                ssim.append(SSIM)
                
                # LPIPS (using torch)
                if lpips_cal is not None:
                    fake_t = to_tensor(fake)
                    real_t = to_tensor(real)
                    d = lpips_cal.calculate(fake_t, real_t)
                    lpips_scores.append(d)
                
            except Exception as e:
                print(f"Error processing {i}: {e}")
        else:
            continue
            
    if len(psnr) > 0:
        average_psnr = sum(psnr) / len(psnr)
        average_ssim = sum(ssim) / len(ssim)
        if len(lpips_scores) > 0:
            average_lpips = sum(lpips_scores) / len(lpips_scores)
        else:
            average_lpips = 0.0
        
        print(f"Results for {result_path}")
        print(f"Average PSNR: {average_psnr:.4f}")
        print(f"Average SSIM: {average_ssim:.4f}")
        print(f"Average LPIPS: {average_lpips:.4f}")
    else:
        print("No images processed.")

if __name__ == '__main__':
    opt = parse_opt()
    evaluate_metrics(opt.result_path, opt.gpu)