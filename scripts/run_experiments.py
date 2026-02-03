#!/usr/bin/env python3
"""
Main experiment runner for BCI transfer learning and HER2 classification.

This script runs all experiments:
1. Pre-training on full dataset (baseline)
2. Fine-tuning on small datasets (10%, 20%, 50%)
3. Training with HER2 classification head
4. Evaluation and metrics collection

Usage:
    python scripts/run_experiments.py --experiment all
    python scripts/run_experiments.py --experiment pretrain
    python scripts/run_experiments.py --experiment finetune --ratio 0.1
    python scripts/run_experiments.py --experiment classify
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ExperimentRunner:
    """Run and manage experiments."""
    
    def __init__(self, args):
        self.args = args
        self.base_dir = Path(__file__).parent.parent
        self.pix2pix_dir = self.base_dir / 'PyramidPix2pix'
        self.experiments_dir = self.base_dir / 'experiments'
        self.checkpoints_dir = self.experiments_dir / 'checkpoints'
        self.logs_dir = self.experiments_dir / 'logs'
        self.results_dir = self.experiments_dir / 'results'
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment configurations
        self.configs = {
            'pretrain': {
                'name': 'bci_pretrain_full',
                'dataroot': str(self.base_dir / 'BCI_dataset'),
                'n_epochs': 50,
                'n_epochs_decay': 50,
                'batch_size': 2,
                'crop_size': 512,
                'pattern': 'L1_L2_L3_L4',
            },
            'finetune_10': {
                'name': 'bci_finetune_10pct',
                'dataroot': str(self.base_dir / 'BCI_dataset_small_10pct'),
                'pretrained_path': str(self.checkpoints_dir / 'bci_pretrain_full' / 'latest_net_G.pth'),
                'n_epochs': 30,
                'n_epochs_decay': 20,
                'batch_size': 4,
                'crop_size': 512,
                'pattern': 'L1_L2_L3_L4',
                'strong_augment': True,
                'finetune_lr_factor': 0.1,
            },
            'finetune_20': {
                'name': 'bci_finetune_20pct',
                'dataroot': str(self.base_dir / 'BCI_dataset_small_20pct'),
                'pretrained_path': str(self.checkpoints_dir / 'bci_pretrain_full' / 'latest_net_G.pth'),
                'n_epochs': 30,
                'n_epochs_decay': 20,
                'batch_size': 4,
                'crop_size': 512,
                'pattern': 'L1_L2_L3_L4',
                'strong_augment': True,
                'finetune_lr_factor': 0.1,
            },
            'finetune_50': {
                'name': 'bci_finetune_50pct',
                'dataroot': str(self.base_dir / 'BCI_dataset_small_50pct'),
                'pretrained_path': str(self.checkpoints_dir / 'bci_pretrain_full' / 'latest_net_G.pth'),
                'n_epochs': 30,
                'n_epochs_decay': 20,
                'batch_size': 4,
                'crop_size': 512,
                'pattern': 'L1_L2_L3_L4',
                'strong_augment': True,
                'finetune_lr_factor': 0.1,
            },
            'no_transfer_10': {
                'name': 'bci_no_transfer_10pct',
                'dataroot': str(self.base_dir / 'BCI_dataset_small_10pct'),
                'n_epochs': 50,
                'n_epochs_decay': 50,
                'batch_size': 4,
                'crop_size': 512,
                'pattern': 'L1_L2_L3_L4',
                'strong_augment': True,
            },
            'classify': {
                'name': 'bci_with_classifier',
                'dataroot': str(self.base_dir / 'BCI_dataset'),
                'n_epochs': 50,
                'n_epochs_decay': 50,
                'batch_size': 2,
                'crop_size': 512,
                'pattern': 'L1_L2_L3_L4',
                'enable_classification': True,
                'lambda_classifier': 0.5,
                'dataset_mode': 'her2_aligned',
            },
        }
    
    def run_command(self, cmd, cwd=None):
        """Run a command and stream output."""
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        if cwd is None:
            cwd = self.pix2pix_dir
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        return process.returncode
    
    def prepare_small_datasets(self):
        """Create small dataset subsets."""
        print("\n" + "="*60)
        print("Creating small dataset subsets...")
        print("="*60)
        
        script_path = self.base_dir / 'scripts' / 'create_small_dataset.py'
        
        for ratio in [0.1, 0.2, 0.5]:
            target_dir = self.base_dir / f'BCI_dataset_small_{int(ratio*100)}pct'
            
            if target_dir.exists():
                print(f"Dataset {target_dir} already exists, skipping...")
                continue
            
            cmd = [
                sys.executable, str(script_path),
                '--source', str(self.base_dir / 'BCI_dataset'),
                '--target', str(target_dir),
                '--ratio', str(ratio),
                '--seed', '42'
            ]
            
            self.run_command(cmd, cwd=self.base_dir)
    
    def combine_datasets(self):
        """Combine HE and IHC images for pix2pix training."""
        print("\n" + "="*60)
        print("Combining HE and IHC images...")
        print("="*60)
        
        combine_script = self.pix2pix_dir / 'datasets' / 'combine_A_and_B.py'
        
        # Datasets to process
        datasets = [
            ('BCI_dataset', 'datasets/BCI'),
            ('BCI_dataset_small_10pct', 'datasets/BCI_small_10pct'),
            ('BCI_dataset_small_20pct', 'datasets/BCI_small_20pct'),
            ('BCI_dataset_small_50pct', 'datasets/BCI_small_50pct'),
        ]
        
        for source_name, target_name in datasets:
            source_dir = self.base_dir / source_name
            target_dir = self.pix2pix_dir / target_name
            
            if not source_dir.exists():
                print(f"Source {source_dir} not found, skipping...")
                continue
            
            if (target_dir / 'train').exists():
                print(f"Combined dataset {target_dir} already exists, skipping...")
                continue
            
            print(f"\nProcessing {source_name}...")
            
            cmd = [
                sys.executable, str(combine_script),
                '--fold_A', str(source_dir / 'HE'),
                '--fold_B', str(source_dir / 'IHC'),
                '--fold_AB', str(target_dir)
            ]
            
            self.run_command(cmd, cwd=self.pix2pix_dir)
    
    def build_train_command(self, config):
        """Build training command from config."""
        cmd = [
            sys.executable, 'train.py',
            '--dataroot', config['dataroot'],
            '--name', config['name'],
            '--checkpoints_dir', str(self.checkpoints_dir),
            '--n_epochs', str(config.get('n_epochs', 50)),
            '--n_epochs_decay', str(config.get('n_epochs_decay', 50)),
            '--batch_size', str(config.get('batch_size', 2)),
            '--crop_size', str(config.get('crop_size', 512)),
            '--preprocess', 'crop',
            '--pattern', config.get('pattern', 'L1_L2_L3_L4'),
            '--save_epoch_freq', '10',
            '--display_id', '-1',  # Disable visdom
        ]
        
        # GPU settings
        if self.args.gpu_ids:
            cmd.extend(['--gpu_ids', self.args.gpu_ids])
        else:
            cmd.extend(['--gpu_ids', '0'])
        
        # Transfer learning options
        if config.get('pretrained_path'):
            cmd.extend(['--pretrained_path', config['pretrained_path']])
        
        if config.get('strong_augment'):
            cmd.append('--strong_augment')
        
        if config.get('finetune_lr_factor'):
            cmd.extend(['--finetune_lr_factor', str(config['finetune_lr_factor'])])
        
        # Classification options
        if config.get('enable_classification'):
            cmd.append('--enable_classification')
            cmd.extend(['--lambda_classifier', str(config.get('lambda_classifier', 1.0))])
        
        if config.get('dataset_mode'):
            cmd.extend(['--dataset_mode', config['dataset_mode']])
        
        return cmd
    
    def run_training(self, experiment_name):
        """Run a training experiment."""
        if experiment_name not in self.configs:
            print(f"Unknown experiment: {experiment_name}")
            return False
        
        config = self.configs[experiment_name]
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"{'='*60}\n")
        
        # Check if pretrained model exists (for fine-tuning)
        if config.get('pretrained_path') and not Path(config['pretrained_path']).exists():
            print(f"Warning: Pretrained model not found at {config['pretrained_path']}")
            print("Please run pre-training first with: --experiment pretrain")
            return False
        
        cmd = self.build_train_command(config)
        return_code = self.run_command(cmd)
        
        return return_code == 0
    
    def run_evaluation(self, experiment_name):
        """Run evaluation for an experiment."""
        if experiment_name not in self.configs:
            print(f"Unknown experiment: {experiment_name}")
            return
        
        config = self.configs[experiment_name]
        model_name = config['name']
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}\n")
        
        # Run test
        cmd = [
            sys.executable, 'test.py',
            '--dataroot', config['dataroot'],
            '--name', model_name,
            '--checkpoints_dir', str(self.checkpoints_dir),
            '--results_dir', str(self.results_dir),
            '--gpu_ids', self.args.gpu_ids or '0',
            '--preprocess', 'none',
        ]
        
        self.run_command(cmd)
        
        # Run evaluation metrics
        results_path = self.results_dir / model_name / 'test_latest'
        if results_path.exists():
            eval_cmd = [
                sys.executable, 'evaluate.py',
                '--result_path', str(results_path)
            ]
            self.run_command(eval_cmd)
    
    def run_all_experiments(self):
        """Run all experiments."""
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print("="*60)
        
        # Step 1: Prepare datasets
        self.prepare_small_datasets()
        self.combine_datasets()
        
        # Step 2: Pre-training on full dataset
        print("\n\n>>> STEP 2: Pre-training on full dataset")
        self.run_training('pretrain')
        
        # Step 3: Fine-tuning experiments
        print("\n\n>>> STEP 3: Fine-tuning on small datasets")
        for ratio in ['10', '20', '50']:
            self.run_training(f'finetune_{ratio}')
        
        # Step 4: No transfer learning baseline
        print("\n\n>>> STEP 4: No transfer learning baseline (10%)")
        self.run_training('no_transfer_10')
        
        # Step 5: Classification experiment
        print("\n\n>>> STEP 5: Training with HER2 classification")
        self.run_training('classify')
        
        # Step 6: Evaluation
        print("\n\n>>> STEP 6: Evaluation")
        for exp_name in self.configs.keys():
            self.run_evaluation(exp_name)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*60)
    
    def run(self):
        """Main entry point."""
        experiment = self.args.experiment
        
        if experiment == 'prepare':
            self.prepare_small_datasets()
            self.combine_datasets()
        
        elif experiment == 'all':
            self.run_all_experiments()
        
        elif experiment == 'pretrain':
            self.prepare_small_datasets()
            self.combine_datasets()
            self.run_training('pretrain')
        
        elif experiment == 'finetune':
            ratio = int(self.args.ratio * 100)
            self.run_training(f'finetune_{ratio}')
        
        elif experiment == 'classify':
            self.run_training('classify')
        
        elif experiment == 'evaluate':
            for exp_name in self.configs.keys():
                self.run_evaluation(exp_name)
        
        else:
            print(f"Unknown experiment: {experiment}")
            print("Available: all, prepare, pretrain, finetune, classify, evaluate")


def main():
    parser = argparse.ArgumentParser(
        description='Run BCI transfer learning experiments'
    )
    parser.add_argument(
        '--experiment', type=str, default='all',
        choices=['all', 'prepare', 'pretrain', 'finetune', 'classify', 'evaluate'],
        help='Experiment to run'
    )
    parser.add_argument(
        '--ratio', type=float, default=0.1,
        help='Dataset ratio for fine-tuning (0.1, 0.2, 0.5)'
    )
    parser.add_argument(
        '--gpu_ids', type=str, default='0',
        help='GPU IDs to use (e.g., "0" or "0,1")'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode with fewer epochs for testing'
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args)
    runner.run()


if __name__ == '__main__':
    main()


