#!/usr/bin/env python3
"""
Create small dataset subsets for transfer learning experiments.
Maintains HER2 status distribution (stratified sampling).

Usage:
    python scripts/create_small_dataset.py --ratio 0.1 --seed 42
    python scripts/create_small_dataset.py --ratio 0.2 --seed 42
    python scripts/create_small_dataset.py --ratio 0.5 --seed 42
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import re


def extract_her2_label(filename):
    """
    Extract HER2 label from filename.
    Format: number_train/test_HER2level.png
    
    Returns:
        str: HER2 level ('0', '1+', '2+', '3+') or None if not found
    """
    stem = Path(filename).stem
    # Match pattern: ends with _0, _1+, _2+, or _3+
    match = re.search(r'_([0-3]\+?)$', stem)
    if match:
        return match.group(1)
    return None


def get_her2_distribution(image_dir):
    """
    Get distribution of HER2 levels in directory.
    
    Returns:
        dict: {her2_level: [list of image paths]}
    """
    distribution = defaultdict(list)
    
    image_dir = Path(image_dir)
    for img_path in image_dir.glob('*.png'):
        her2_level = extract_her2_label(img_path.name)
        if her2_level is not None:
            distribution[her2_level].append(img_path)
    
    return distribution


def create_small_dataset(source_dir, target_dir, ratio, seed=42):
    """
    Create stratified small dataset subset.
    
    Args:
        source_dir: Path to original BCI_dataset
        target_dir: Path to output small dataset
        ratio: Fraction of data to keep (0.1 = 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    print(f"\n{'='*60}")
    print(f"Creating {ratio*100:.0f}% subset of BCI dataset")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Process train and test splits
    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Get HER2 distribution for HE images
        he_dir = source_dir / 'HE' / split
        distribution = get_her2_distribution(he_dir)
        
        # Print original distribution
        print(f"\nOriginal distribution ({split}):")
        total_original = 0
        for level in ['0', '1+', '2+', '3+']:
            count = len(distribution.get(level, []))
            total_original += count
            print(f"  HER2 {level}: {count}")
        print(f"  Total: {total_original}")
        
        # Sample stratified subset
        sampled = []
        print(f"\nSampled distribution ({ratio*100:.0f}%):")
        total_sampled = 0
        for level in ['0', '1+', '2+', '3+']:
            files = distribution.get(level, [])
            n_samples = max(1, int(len(files) * ratio))
            selected = random.sample(files, min(n_samples, len(files)))
            sampled.extend(selected)
            total_sampled += len(selected)
            print(f"  HER2 {level}: {len(selected)}")
        print(f"  Total: {total_sampled}")
        
        # Copy files
        for he_path in sampled:
            # Create target directories
            target_he_dir = target_dir / 'HE' / split
            target_ihc_dir = target_dir / 'IHC' / split
            target_he_dir.mkdir(parents=True, exist_ok=True)
            target_ihc_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy HE image
            target_he = target_he_dir / he_path.name
            shutil.copy2(he_path, target_he)
            
            # Copy corresponding IHC image
            ihc_path = source_dir / 'IHC' / split / he_path.name
            if ihc_path.exists():
                target_ihc = target_ihc_dir / he_path.name
                shutil.copy2(ihc_path, target_ihc)
            else:
                print(f"  Warning: IHC image not found: {ihc_path}")
    
    # Create README
    readme_path = target_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(f"BCI Dataset - {ratio*100:.0f}% Subset\n")
        f.write(f"{'='*40}\n\n")
        f.write(f"Created with seed: {seed}\n")
        f.write(f"Ratio: {ratio}\n\n")
        f.write("This is a stratified subset maintaining HER2 distribution.\n")
        f.write("Original dataset: https://github.com/bupt-ai-cz/BCI\n")
    
    print(f"\n{'='*60}")
    print(f"Dataset created successfully at: {target_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create small dataset subset for transfer learning'
    )
    parser.add_argument(
        '--source', type=str, 
        default='./BCI_dataset',
        help='Path to original BCI_dataset'
    )
    parser.add_argument(
        '--target', type=str, 
        default=None,
        help='Path to output dataset (default: BCI_dataset_small_{ratio}pct)'
    )
    parser.add_argument(
        '--ratio', type=float, 
        default=0.1,
        help='Fraction of data to keep (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--seed', type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Create all subsets (10%%, 20%%, 50%%)'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Create all standard subsets
        for ratio in [0.1, 0.2, 0.5]:
            target = f"./BCI_dataset_small_{int(ratio*100)}pct"
            create_small_dataset(args.source, target, ratio, args.seed)
    else:
        target = args.target
        if target is None:
            target = f"./BCI_dataset_small_{int(args.ratio*100)}pct"
        create_small_dataset(args.source, target, args.ratio, args.seed)


if __name__ == '__main__':
    main()


