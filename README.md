# CS6886W: Systems Engineering for Deep Learning — Assignment 2
## In-Memory Model Compression of MobileNet-v2 on CIFAR-10

- **Student Name:** Kiran T K
- **Roll Number:** `cs25m513`
- **Course:** CS6886W — Systems Engineering for Deep Learning, IIT Madras
- **GitHub Repository:** [https://github.com/cs25m513/CS6886W-Assignment2-MobileNetV2-](https://github.com/cs25m513/CS6886W-Assignment2-MobileNetV2-)

This repository contains the complete, modular implementation for **CS6886W Assignment 2**: training an adapted **MobileNet-v2** architecture on **CIFAR-10** and compressing both **weights and activations** in RAM/VRAM using a custom, from-scratch compression pipeline (**no external compression APIs or `torch.ao.quantization` libraries used**).

---

## 1. Repository Structure (Question 5a)

```text
cs6886w_assignment2/
├── README.md                        # Exact commands, environment, dependency versions, seeds (Q5b)
├── requirements.txt                 # Pinned Python package versions
├── config.py                        # Deterministic seed setup & training hyperparameters
├── data.py                          # CIFAR-10 normalization, RandomCrop, Flip, & Cutout (Q1a)
├── models/
│   ├── __init__.py
│   └── mobilenetv2.py               # CIFAR-10 adapted MobileNet-v2 (32x32 resolution preservation) (Q1b)
├── compression/
│   ├── __init__.py
│   ├── bitpack.py                   # Vectorized sub-byte (2/3/4/6/8-bit) uint8 tensor packing/unpacking (Q2a)
│   ├── quantizer.py                 # BatchNorm folding + Per-Channel Weight & Activation Quantizers (Q2a, Q2b)
│   └── memory_profiler.py           # Exact in-memory byte accounting for weights, metadata & activations (Q2c, Q4)
├── train.py                         # Baseline training script with Warmup + Cosine Annealing LR (Q1)
├── test.py                          # Required CLI entry point: python test.py --weight_quant_bits 8 --activation_quant_bits 8
├── sweep.py                         # 13-run compression sweep & WandB Parallel Coordinates logger (Q3)
├── generate_plots.py                # Generates loss/accuracy curves, parallel coordinates & Pareto plots
├── CS6886W_Assignment2_Colab.ipynb  # Self-contained Google Colab GPU notebook
├── results/
│   ├── training_history.csv         # Epoch-by-epoch baseline training logs
│   ├── compression_sweep.csv        # 13-run quantization sweep metrics & memory breakdown
└── report/
    ├── main.tex                     # LaTeX source of the final assignment report
    └── CS6886W_Assignment2_Report.pdf # Compiled PDF report for submission
```

---

## 2. Environment & Dependency Versions (Question 5b)

- **Operating System**: Linux (Ubuntu 22.04+ / Debian / Google Colab GPU Runtime)
- **Python Version**: `3.10+` (tested on `3.11` / `3.13`)
- **Pinned Dependencies** (`requirements.txt`):
  - `torch==2.5.1`
  - `torchvision==0.20.1`
  - `numpy==1.26.4`
  - `matplotlib==3.9.2`
  - `pandas==2.2.3`
  - `wandb==0.18.7`
  - `tqdm==4.67.0`

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Deterministic Seed Configuration (Question 5b)

All scripts (`train.py`, `test.py`, `sweep.py`) invoke `set_seed(seed=42)` defined in `config.py`:
```python
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 4. Exact Commands to Reproduce Results

### Step 1: Train the Baseline MobileNet-v2 Model on CIFAR-10 (Question 1)
```bash
python train.py \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.05 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --label_smoothing 0.1 \
    --warmup_epochs 5 \
    --seed 42
```
- Saves best checkpoint to `./checkpoints/mobilenetv2_cifar10_best.pth` (**94.28% Top-1 Test Accuracy**).
- Logs per-epoch metrics to `./results/training_history.csv`.

### Step 2: Evaluate & Compress MobileNet-v2 at Any Bit-Width (Question 2 & Question 4)
Matches the exact command-line interface specified in the assignment instructions:
```bash
# 8-bit weights, 8-bit activations (W8A8)
python test.py --weight_quant_bits 8 --activation_quant_bits 8

# 6-bit weights, 6-bit activations (W6A6)
python test.py --weight_quant_bits 6 --activation_quant_bits 6

# 4-bit weights, 4-bit activations (W4A4)
python test.py --weight_quant_bits 4 --activation_quant_bits 4
```

### Step 3: Run the 13-Configuration Compression Sweep & WandB Parallel Coordinates Chart (Question 3)
```bash
# Run sweep locally and save to ./results/compression_sweep.csv
python sweep.py --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth --seed 42

# Optional: Log directly to Weights & Biases for interactive Parallel Coordinates Chart
wandb login
python sweep.py --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth --use_wandb --wandb_project cs6886w-assignment2-sweep

# Generate all report plots
python generate_plots.py
```

---

## 5. Summary of Results

| Configuration | Weight Bits | Act Bits | Top-1 Test Acc (%) | Acc Drop (%) | Packed Weights (MB) | Metadata Overhead (KB) | Final Model Size (MB) | Weight Comp. Ratio | Model Comp. Ratio | Act Comp. Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 32 | 32 | **94.28%** | 0.00% | 8.4021 | 266.50 (BN) | **8.6624** | 1.00x | 1.00x | 1.00x |
| **W8A8** | 8 | 8 | **94.19%** | -0.09% | 2.1005 | 83.84 | **2.1824** | 3.91x | 3.97x | 4.00x |
| **W6A6** | 6 | 6 | **93.74%** | -0.54% | 1.5754 | 83.84 | **1.6573** | 5.17x | 5.23x | 5.33x |
| **W4A8** | 4 | 8 | **92.65%** | -1.63% | 1.0503 | 83.84 | **1.1321** | 7.64x | 7.65x | 4.00x |
| **W4A4 (Sweet Spot)** | 4 | 4 | **91.12%** | -3.16% | 1.0503 | 83.84 | **1.1321** | 7.64x | **7.65x** | **8.00x** |
| **W3A4** | 3 | 4 | **86.15%** | -8.13% | 0.7877 | 83.84 | **0.8696** | 10.04x | **9.96x** | 8.00x |
| **W2A2 (Max Comp.)** | 2 | 2 | **38.60%** | -55.68% | 0.5251 | 83.84 | **0.6070** | **14.64x** | **14.27x** | **15.99x** |
