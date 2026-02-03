#!/bin/bash
# Quick start script for BCI Transfer Learning experiments
# Usage: bash scripts/quick_start.sh

set -e  # Exit on error

echo "=============================================="
echo "BCI Transfer Learning Pipeline - Quick Start"
echo "=============================================="

# Check if we're in the right directory
if [ ! -d "BCI_dataset" ]; then
    echo "Error: BCI_dataset not found!"
    echo "Please run this script from the BCI-main directory"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "[Step 1/5] Installing dependencies..."
pip install -r PyramidPix2pix/requirements.txt
pip install -r requirements_extra.txt

# Step 2: Create small datasets
echo ""
echo "[Step 2/5] Creating small datasets (10%, 20%, 50%)..."
python scripts/create_small_dataset.py --all

# Step 3: Combine datasets for pix2pix format
echo ""
echo "[Step 3/5] Combining HE and IHC images..."
cd PyramidPix2pix

# Full dataset
if [ ! -d "datasets/BCI/train" ]; then
    python datasets/combine_A_and_B.py \
        --fold_A ../BCI_dataset/HE \
        --fold_B ../BCI_dataset/IHC \
        --fold_AB datasets/BCI
fi

# Small datasets
for pct in 10 20 50; do
    if [ -d "../BCI_dataset_small_${pct}pct" ] && [ ! -d "datasets/BCI_small_${pct}pct/train" ]; then
        python datasets/combine_A_and_B.py \
            --fold_A ../BCI_dataset_small_${pct}pct/HE \
            --fold_B ../BCI_dataset_small_${pct}pct/IHC \
            --fold_AB datasets/BCI_small_${pct}pct
    fi
done

cd ..

# Step 4: Quick training test (reduced epochs)
echo ""
echo "[Step 4/5] Running quick training test (10 epochs)..."
cd PyramidPix2pix

python train.py \
    --dataroot ./datasets/BCI \
    --name bci_quick_test \
    --gpu_ids 0 \
    --pattern L1_L2_L3_L4 \
    --batch_size 2 \
    --crop_size 512 \
    --preprocess crop \
    --n_epochs 5 \
    --n_epochs_decay 5 \
    --save_epoch_freq 5 \
    --display_id -1

cd ..

# Step 5: Test
echo ""
echo "[Step 5/5] Running test..."
cd PyramidPix2pix

python test.py \
    --dataroot ./datasets/BCI \
    --name bci_quick_test \
    --gpu_ids 0 \
    --preprocess none \
    --num_test 20

cd ..

echo ""
echo "=============================================="
echo "Quick start completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Run full experiments: python scripts/run_experiments.py --experiment all"
echo "2. Analyze results: jupyter notebook notebooks/analysis.ipynb"
echo ""


