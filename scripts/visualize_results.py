import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.options.test_options import TestOptions
from src.data import create_dataset
from src.models import create_model
from src.utils import util

def get_her2_class(filepath):
    """Extract HER2 class from filename (e.g., '00262_train_3+.png' -> '3+')"""
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if len(parts) >= 3:
        # Extract the part after 'train' or 'test'
        # Example: 00000_train_1+.png -> 1+
        # Example: 00000_test_1+.png -> 1+
        label_part = parts[-1].replace('.png', '')
        return label_part
    return "Unknown"

def compute_brownness(image_tensor):
    """
    Compute heuristic brownness score.
    Brown is roughly High Red, Low/Medium Green, Low Blue.
    Simple heuristic: R > G > B.
    We can compute mean intensity of (R - B) + (R - G) in normalized range.
    Input: tensor [-1, 1], shape (C, H, W)
    """
    # Convert to [0, 1]
    img = (image_tensor + 1) / 2.0
    r = img[0, :, :]
    g = img[1, :, :]
    b = img[2, :, :]
    
    # Simple brownness index: R relative to B and G
    # DAB stain is brown. Hematoxylin is blue.
    # Brown area has high R, lower B. Blue area has high B.
    # We can measure mean (R - B).
    brownness = torch.mean(r - b).item()
    return brownness

def visualize():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.eval = True       # use eval mode

    # Create dataset & model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    output_dir = os.path.join(opt.results_dir, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations in {output_dir}...")

    font_path = "/System/Library/Fonts/Helvetica.ttc" # Mac default font
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
            
        model.set_input(data)
        model.test()
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()[0]
        her2_class = get_her2_class(img_path)
        
        # Unpack images (Real A is HE, Real B is IHC, Fake B is Generated IHC)
        # Note: 'real_A' might be input (HE) and 'real_B' target (IHC) depending on direction
        # config.yaml says direction: "AtoB" # A = HE, B = IHC
        
        real_A = util.tensor2im(visuals['real_A'])
        fake_B = util.tensor2im(visuals['fake_B'])
        real_B = util.tensor2im(visuals['real_B'])
        
        # Calculate brownness on tensors (still in roughly [-1,1] range in model.fake_B)
        # We need access to the tensors. model.fake_B is the tensor.
        brownness_score = compute_brownness(model.fake_B[0].cpu())
        
        # Create composite image
        # Horizontal stack: Real HE | Real IHC | Fake IHC
        imgs_comb = np.hstack((real_A, real_B, fake_B))
        imgs_pil = Image.fromarray(imgs_comb)
        
        draw = ImageDraw.Draw(imgs_pil)
        
        # Annotate
        # Add a white band at top for text? Or just overlay?
        # Let's add top border
        border_size = 40
        new_img = Image.new('RGB', (imgs_pil.width, imgs_pil.height + border_size), (255, 255, 255))
        new_img.paste(imgs_pil, (0, border_size))
        draw = ImageDraw.Draw(new_img)
        
        info_text = f"HER2 Class: {her2_class} | Brownness (Fake): {brownness_score:.3f}"
        
        # Column labels
        w = real_A.shape[1]
        draw.text((10, 10), info_text, fill=(0, 0, 0), font=font)
        draw.text((10, border_size + 5), "Input HE", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0,0,0), font=font)
        draw.text((w + 10, border_size + 5), "Real IHC", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0,0,0), font=font)
        draw.text((2*w + 10, border_size + 5), "Fake IHC", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0,0,0), font=font)

        out_name = f"{her2_class}_{os.path.basename(img_path)}"
        save_path = os.path.join(output_dir, out_name)
        new_img.save(save_path)
        
        if i % 10 == 0:
            print(f"Processed {i} images...")

    print("Visualization complete.")

if __name__ == '__main__':
    visualize()
