"""
Dataset class for paired HE-IHC images with HER2 classification labels.
Supports strong augmentation for small dataset training.
"""

import os
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from src.data.base_dataset import BaseDataset, get_params, get_transform
from src.data.image_folder import make_dataset
from src.utils.her2_utils import get_her2_label_from_path, get_class_weights


class HER2AlignedDataset(BaseDataset):
    """
    Dataset class for paired HE-IHC images with HER2 labels.
    Supports stratified sampling and strong augmentation.
    """
    
    def __init__(self, opt):
        """Initialize dataset with HER2 labels."""
        BaseDataset.__init__(self, opt)
        
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        
        assert(self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        # Extract HER2 labels
        self.her2_labels = []
        for path in self.AB_paths:
            label = get_her2_label_from_path(path)
            self.her2_labels.append(label)
        
        # Filter out invalid labels if needed
        valid_indices = [i for i, l in enumerate(self.her2_labels) if l >= 0]
        if len(valid_indices) < len(self.AB_paths):
            print(f"Warning: {len(self.AB_paths) - len(valid_indices)} images with unknown HER2 status")
        
        # Calculate class weights for imbalanced learning
        self.class_weights = torch.tensor(
            get_class_weights([l for l in self.her2_labels if l >= 0]),
            dtype=torch.float32
        )
        
        # Strong augmentation flag
        self.strong_augment = getattr(opt, 'strong_augment', False)
        
        # Print dataset info
        self._print_dataset_info()
    
    def _print_dataset_info(self):
        """Print dataset statistics."""
        from collections import Counter
        counter = Counter(self.her2_labels)
        print(f"\nHER2AlignedDataset ({self.opt.phase}):")
        print(f"  Total images: {len(self.AB_paths)}")
        for label in [0, 1, 2, 3]:
            count = counter.get(label, 0)
            print(f"  HER2 {['0', '1+', '2+', '3+'][label]}: {count}")
        if counter.get(-1, 0) > 0:
            print(f"  Unknown: {counter[-1]}")
    
    def __getitem__(self, index):
        """Return data point with HER2 label."""
        # Read image pair
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        
        # Get HER2 label
        her2_label = self.her2_labels[index]
        
        # Apply transforms
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        A = A_transform(A)
        B = B_transform(B)
        
        # Apply strong augmentation if enabled
        if self.strong_augment and self.opt.isTrain:
            A, B = self._apply_strong_augmentation(A, B)
        
        return {
            'A': A,
            'B': B,
            'A_paths': AB_path,
            'B_paths': AB_path,
            'her2_label': torch.tensor(her2_label, dtype=torch.long)
        }
    
    def _apply_strong_augmentation(self, A, B):
        """
        Apply strong augmentation for small dataset training.
        Same transform applied to both A and B to maintain alignment.
        """
        # Random seed for synchronized transforms
        seed = random.randint(0, 2**32)
        
        # Color jitter (different for A and B as they have different staining)
        if random.random() > 0.5:
            # Moderate color jitter for HE
            random.seed(seed)
            torch.manual_seed(seed)
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            
            A = TF.adjust_brightness(A, brightness)
            A = TF.adjust_contrast(A, contrast)
            A = TF.adjust_saturation(A, saturation)
            
            # Similar for IHC
            B = TF.adjust_brightness(B, brightness)
            B = TF.adjust_contrast(B, contrast)
            B = TF.adjust_saturation(B, saturation)
        
        # Random affine (synchronized)
        if random.random() > 0.5:
            random.seed(seed)
            torch.manual_seed(seed)
            angle = random.uniform(-10, 10)
            translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-5, 5)
            
            # Convert translate to pixels
            _, h, w = A.shape
            translate_px = (int(translate[0] * w), int(translate[1] * h))
            
            A = TF.affine(A, angle=angle, translate=translate_px, scale=scale, shear=shear)
            B = TF.affine(B, angle=angle, translate=translate_px, scale=scale, shear=shear)
        
        # Random Gaussian blur
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            A = TF.gaussian_blur(A, kernel_size=kernel_size, sigma=sigma)
            B = TF.gaussian_blur(B, kernel_size=kernel_size, sigma=sigma)
        
        # Add Gaussian noise
        if random.random() > 0.7:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(A) * noise_std
            A = torch.clamp(A + noise, -1, 1)
            B = torch.clamp(B + noise, -1, 1)
        
        return A, B
    
    def __len__(self):
        """Return dataset size."""
        return len(self.AB_paths)
    
    def get_class_weights(self):
        """Return class weights for weighted loss."""
        return self.class_weights


class HER2ClassificationDataset(BaseDataset):
    """
    Dataset for HER2 classification only (IHC or generated images).
    """
    
    def __init__(self, opt, image_dir=None, generated=False):
        """
        Initialize classification dataset.
        
        Args:
            opt: Options
            image_dir: Optional custom image directory
            generated: If True, load from generated images directory
        """
        BaseDataset.__init__(self, opt)
        
        if image_dir:
            self.image_dir = image_dir
        elif generated:
            self.image_dir = os.path.join(opt.results_dir, opt.name, 'generated')
        else:
            self.image_dir = os.path.join(opt.dataroot, 'IHC', opt.phase)
        
        # Get image paths
        self.image_paths = sorted([
            os.path.join(self.image_dir, f) 
            for f in os.listdir(self.image_dir) 
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        
        # Extract labels
        self.labels = [get_her2_label_from_path(p) for p in self.image_paths]
        
        # Filter valid
        valid = [(p, l) for p, l in zip(self.image_paths, self.labels) if l >= 0]
        self.image_paths = [p for p, _ in valid]
        self.labels = [l for _, l in valid]
        
        # Class weights
        self.class_weights = torch.tensor(
            get_class_weights(self.labels),
            dtype=torch.float32
        )
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = self.labels[index]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': path
        }
    
    def __len__(self):
        return len(self.image_paths)


