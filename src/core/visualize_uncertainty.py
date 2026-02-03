import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.options.test_options import TestOptions
from src.data import create_dataset
from src.models import create_model
from src.utils import util

def visualize_uncertainty():
    opt = TestOptions().parse()
    # Hard-code parameters for uncertainty estimation
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    # Enable MC Dropout: Ensure model is in train mode (or specifically dropout is on)
    # We do NOT call model.eval() or if we do, we must maintain dropout.
    # By default, not calling model.eval() keeps it in train mode.
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    # We want to use dropout, so ensure we are in train mode or force dropout
    model.netG.train() 
    
    save_dir = os.path.join(opt.results_dir, opt.name, 'uncertainty')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating uncertainty maps in {save_dir}...")
    
    N_SAMPLES = 20
    
    for i, data in enumerate(tqdm(dataset)):
        if i >= opt.num_test:
            break
            
        model.set_input(data)
        img_path = model.get_image_paths()[0]
        short_path = os.path.basename(img_path)
        name = os.path.splitext(short_path)[0]
        
        preds = []
        for _ in range(N_SAMPLES):
            model.test() # This typically calls forward(). Check if model.test() calls eval(). 
            # Pix2PixModel.test() just calls forward().
            # BUT BaseOptions sets model.eval() if opt.eval is set. We didn't set it.
            # However, torch.no_grad() might be used in test().
            # Let's check model.test() implementation. 
            # If it uses torch.no_grad(), proper dropout might still work if module is in train mode.
            
            visuals = model.get_current_visuals()
            # fake_B is the prediction. shape (1, C, H, W). Range [-1, 1] usually.
            pred = visuals['fake_B'].cpu().float().numpy()
            preds.append(pred)
            
        preds = np.concatenate(preds, axis=0) # (N, C, H, W)
        
        # Calculate Mean and Std
        mean_pred = np.mean(preds, axis=0) # (C, H, W)
        std_pred = np.std(preds, axis=0)   # (C, H, W) - Uncertainty per channel
        
        # Uncertainty map: Average std across channels or L2 norm?
        # For RGB, average std is a simple metric.
        uncertainty_map = np.mean(std_pred, axis=0) # (H, W)
        
        # Save Mean Prediction
        mean_img = util.tensor2im(torch.from_numpy(mean_pred).unsqueeze(0))
        util.save_image(mean_img, os.path.join(save_dir, f"{name}_mean.png"))
        
        # Save Uncertainty Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(uncertainty_map, cmap='viridis', cbar=True)
        plt.axis('off')
        plt.title(f"Uncertainty: {name}")
        plt.savefig(os.path.join(save_dir, f"{name}_uncertainty.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Also save Real B for reference
        real_B = util.tensor2im(data['B'])
        util.save_image(real_B, os.path.join(save_dir, f"{name}_real.png"))

if __name__ == '__main__':
    visualize_uncertainty()
