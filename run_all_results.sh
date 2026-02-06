#!/bin/bash

# BCI Project - Commands to Show Results to Supervisor
# =====================================================

cd "/Users/marleshkaa./Library/Mobile Documents/com~apple~CloudDocs/Documents/Work/RA/Work with Tomiris/BCI-main"

echo "========================================="
echo "1. Generating Loss Plots"
echo "========================================="
python3 scripts/plot_losses.py \
    --log_path "PyramidPix2pix/checkpoints/bci_demo_10pct/loss_log.txt" \
    --output_path "results/training_losses.png"

echo ""
echo "========================================="
echo "2. Creating Visualizations (100 samples)"
echo "   Time: ~10-15 minutes"
echo "========================================="
PYTHONPATH=. python3 scripts/visualize_results.py \
    --dataroot ./datasets/BCI \
    --name trained_on_BCI \
    --model pix2pix \
    --netG attention_unet_32 \
    --num_test 100 \
    --gpu_ids -1

echo ""
echo "========================================="
echo "✅ ALL DONE!"
echo "========================================="
echo "Results are in:"
echo "  - Loss plots: results/training_losses.png"
echo "  - Visualizations: results/visualization/"
echo ""
