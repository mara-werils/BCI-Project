#!/usr/bin/env python3
"""
Evaluate all experiments and generate summary report.

Usage:
    python scripts/evaluate_all.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'PyramidPix2pix'))

try:
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Please install: pip install scikit-image Pillow")
    sys.exit(1)


class ExperimentEvaluator:
    """Evaluate all experiments and generate report."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / 'experiments' / 'results'
        self.checkpoints_dir = self.base_dir / 'experiments' / 'checkpoints'
        self.data_dir = self.base_dir / 'BCI_dataset'
        
        self.experiments = [
            ('Full Dataset (Baseline)', 'bci_pretrain_full'),
            ('Transfer 10%', 'bci_finetune_10pct'),
            ('Transfer 20%', 'bci_finetune_20pct'),
            ('Transfer 50%', 'bci_finetune_50pct'),
            ('No Transfer 10%', 'bci_no_transfer_10pct'),
            ('With Classification', 'bci_with_classifier'),
        ]
    
    def calculate_metrics(self, exp_name, max_images=None):
        """Calculate PSNR and SSIM for experiment."""
        results_path = self.results_dir / exp_name / 'test_latest' / 'images'
        
        if not results_path.exists():
            return None
        
        psnr_values = []
        ssim_values = []
        
        # Find fake images
        fake_images = sorted(list(results_path.glob('*_fake_B.png')))
        
        if max_images:
            fake_images = fake_images[:max_images]
        
        for fake_path in fake_images:
            # Get corresponding real image
            base_name = fake_path.stem.replace('_fake_B', '')
            real_path = results_path / f"{base_name}_real_B.png"
            
            if not real_path.exists():
                continue
            
            try:
                fake_img = np.array(Image.open(fake_path).convert('RGB'))
                real_img = np.array(Image.open(real_path).convert('RGB'))
                
                if fake_img.shape != real_img.shape:
                    h = min(fake_img.shape[0], real_img.shape[0])
                    w = min(fake_img.shape[1], real_img.shape[1])
                    fake_img = fake_img[:h, :w]
                    real_img = real_img[:h, :w]
                
                psnr_val = psnr(real_img, fake_img, data_range=255)
                ssim_val = ssim(real_img, fake_img, channel_axis=2, data_range=255)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
            except Exception as e:
                print(f"Error processing {fake_path}: {e}")
                continue
        
        if not psnr_values:
            return None
        
        return {
            'psnr_mean': float(np.mean(psnr_values)),
            'psnr_std': float(np.std(psnr_values)),
            'ssim_mean': float(np.mean(ssim_values)),
            'ssim_std': float(np.std(ssim_values)),
            'n_images': len(psnr_values)
        }
    
    def load_classification_results(self, exp_name):
        """Load classification predictions if available."""
        pred_path = self.checkpoints_dir / exp_name / 'predictions.json'
        
        if pred_path.exists():
            with open(pred_path) as f:
                return json.load(f)
        return None
    
    def evaluate_all(self):
        """Evaluate all experiments."""
        print("\n" + "="*60)
        print("EVALUATING ALL EXPERIMENTS")
        print("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'experiments': {}
        }
        
        for name, exp_name in self.experiments:
            print(f"\nEvaluating: {name} ({exp_name})")
            print("-" * 40)
            
            exp_results = {
                'name': name,
                'generation': None,
                'classification': None
            }
            
            # Generation metrics
            metrics = self.calculate_metrics(exp_name)
            if metrics:
                exp_results['generation'] = metrics
                print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
                print(f"  SSIM: {metrics['ssim_mean']:.3f} ± {metrics['ssim_std']:.3f}")
                print(f"  Images: {metrics['n_images']}")
            else:
                print("  No generation results found")
            
            # Classification metrics
            class_results = self.load_classification_results(exp_name)
            if class_results:
                exp_results['classification'] = class_results
                if 'accuracy' in class_results:
                    print(f"  Accuracy: {class_results['accuracy']:.3f}")
            
            results['experiments'][exp_name] = exp_results
        
        # Save results
        output_path = self.base_dir / 'experiments' / 'evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print(f"Results saved to: {output_path}")
        print("="*60)
        
        # Generate summary table
        self.print_summary_table(results)
        
        return results
    
    def print_summary_table(self, results):
        """Print summary table in markdown format."""
        print("\n## Summary Table (for paper)")
        print()
        print("| Method | PSNR ↑ | SSIM ↑ | N Images |")
        print("|--------|--------|--------|----------|")
        
        for exp_name, exp_data in results['experiments'].items():
            name = exp_data['name']
            gen = exp_data.get('generation')
            
            if gen:
                psnr_str = f"{gen['psnr_mean']:.2f} ± {gen['psnr_std']:.2f}"
                ssim_str = f"{gen['ssim_mean']:.3f} ± {gen['ssim_std']:.3f}"
                n_str = str(gen['n_images'])
            else:
                psnr_str = ssim_str = n_str = "N/A"
            
            print(f"| {name} | {psnr_str} | {ssim_str} | {n_str} |")


def main():
    evaluator = ExperimentEvaluator()
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()


