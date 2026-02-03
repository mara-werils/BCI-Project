# BCI: Breast Cancer Immunohistochemical Image Generation
## Enhanced Research Repository

![Status](https://img.shields.io/badge/Status-Research_Grade-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

**Original Paper**: [BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix](https://arxiv.org/pdf/2204.11425v1.pdf)  
**Enhanced Implementation**: This repository contains a scientifically rigorous implementation of the BCI framework, augmented with Uncertainty Quantification and Perceptual Metrics.

---

## 🔬 Scientific Enhancements ("The 1000x Value")

### 1. Uncertainty Quantification (Clinical Trust)
In medical imaging, a generative model must be trustworthy. We implement **Monte Carlo Dropout** inference to generate **Uncertainty Maps**.
- **Dark Areas**: High confidence (Model is sure).
- **Bright Areas**: High specific uncertainty (Model is hallucinating or unsure about tissue details).
This allows pathologists to identify regions where the synthetic IHC image might be unreliable.

### 2. Advanced Perceptual Metrics
Standard PSNR/SSIM metrics favor blurry images. We integrate **LPIPS (Learned Perceptual Image Patch Similarity)** to measure texture fidelity, which is critical for identifying cancerous cells.

### 3. Reproducible Engineering
- **Config-based Experiments**: `configs/config.yaml` replaces messy command-line arguments.
- **Package Structure**: Modular `src/` layout for better maintainability.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/BCI-Enhanced.git
cd BCI-Enhanced
pip install -r requirements.txt
```

### Training
Train the model using the configuration file:
```bash
python train_net.py --config configs/config.yaml
```

### Evaluation (New Metrics)
Calculate PSNR, SSIM, and **LPIPS**:
```bash
python src/core/evaluate.py --result_path ./results/pyramidpix2pix --gpu
```

### Uncertainty Visualization
Generate Uncertainty Maps for the test set:
```bash
python src/core/visualize_uncertainty.py --dataroot ./datasets/BCI --name bci_experiment_v1
```
Results will be saved in `results/bci_experiment_v1/uncertainty/`.

---

## 📂 Repository Structure

```
├── configs/             # Configuration files
├── src/
│   ├── core/            # Core scripts (train, evaluate, uncertainty)
│   ├── data/            # Dataset loading logic
│   ├── metrics/         # Scientific metrics (LPIPS)
│   ├── models/          # PyTorch models (Pyramid Pix2pix)
│   ├── options/         # Argument parsers
│   └── utils/           # Helper utilities
├── tests/               # Automated tests
└── train_net.py         # Main entry point
```

## 📜 Citation

If you use this enhanced repository, please cite the original paper and this implementation:

```bibtex
@inproceedings{Liu_2022_CVPR,
    author    = {Liu, Shengjie and Zhu, Chuang and Xu, Feng and Jia, Xinyu and Shi, Zhongyue and Jin, Mulan},
    title     = {BCI: Breast Cancer Immunohistochemical Image Generation Through Pyramid Pix2pix},
    booktitle = {CVPR Workshops},
    year      = {2022}
}
```
