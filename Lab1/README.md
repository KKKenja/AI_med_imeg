# Lab1 - Pneumonia Classification

Classifies chest X-ray images as **NORMAL** or **PNEUMONIA** with ImageNet-pretrained ResNet18 / ResNet50 (torchvision); the final `fc` layer is re-initialized for 2 classes.

Dataset: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Results

| Model | Batch size | Best Val Acc (epoch) | Test Acc | Test F1 |
|---|---|---|---|---|
| ResNet18 | 64 | 98.46% (22) | 96.92% | 0.9787 |
| ResNet50 | 32 | 98.12% (29) | **98.12%** | **0.9871** |

The test set is evaluated **only once**, using the checkpoint with the highest validation accuracy.

## Files

| File | Description |
|---|---|
| `split_dataset.py` | Pools the original train/val/test images and re-splits them 80/10/10 per class (seed 42) into `data/` |
| `train.py` | Training: dataloaders, data augmentation, model, train/val loop, final test, plots, log file |
| `inference.py` | Loads a saved checkpoint and evaluates it on the test set |
| `plot_compare.py` | Plots the accuracy / F1 curves of several models from their training logs |
| `logs/` | Training logs (all terminal output, plus the arguments of each run) |

## 1. Environment

Python 3.12, PyTorch 2.6 (CUDA 12.4). Tested on an RTX 3050 Ti Laptop GPU (4 GB).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib seaborn tqdm
```

## 2. Prepare the dataset

Download and unzip the Kaggle dataset to `raw/archive/chest_xray/`:

```
raw/archive/chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}
```

The official `val` split has only 16 images, so we re-split everything:

```bash
python split_dataset.py
```

This **copies** the images into `data/` (`raw/` is not modified):

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| train | 1267 | 3419 | 4686 |
| val | 158 | 427 | 585 |
| test | 158 | 427 | 585 |

## 3. Training

```bash
python train.py --model resnet18 --batch_size 64
python train.py --model resnet50 --batch_size 32
```

Each run writes:

| Output | Description |
|---|---|
| `output/<model>_weights.pt` | Weights of the epoch with the highest val accuracy (change with `--ckpt`) |
| `output/<model>_accuracy.png` | Train / val accuracy curves |
| `output/<model>_f1_score.png` | Val F1-score curve |
| `output/<model>_confusion_matrix.png` | Confusion matrix on the test set |
| `logs/<model>_train_<time>.log` | Full training log |

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet18` | `resnet18` or `resnet50` |
| `--num_epochs` | 30 | |
| `--batch_size` | 32 | Use 32 for ResNet50 on a 4 GB GPU |
| `--lr` | 1e-5 | Adam learning rate |
| `--wd` | 1e-4 | Weight decay |
| `--degree` | 15 | Random rotation range (±degrees) |
| `--resize` | 224 | Input size |
| `--dataset` | `data` | Dataset folder created by `split_dataset.py` |
| `--output_dir` | `output` | Where weights and figures are saved |
| `--ckpt` | `output/<model>_weights.pt` | Where the best weights are saved |

Training details:
- Data loading: a custom `ChestXrayDataset` (subclass of `torch.utils.data.Dataset`) scans `data/<split>/{NORMAL,PNEUMONIA}`, labels NORMAL = 0 / PNEUMONIA = 1, converts every image to 3-channel RGB, and is batched with the PyTorch `DataLoader`
- Data augmentation (train only): Resize 224 → RandomHorizontalFlip(p=0.25) → RandomRotation(±15°) → RandomAffine (translate 5%, scale 0.9–1.1) → ColorJitter (brightness / contrast 0.2) → Normalize (ImageNet mean/std)
- Val / test: Resize 224 → Normalize only
- Loss: CrossEntropyLoss with class weights `N / count_c`, to handle the ~1:2.7 NORMAL:PNEUMONIA imbalance
- Optimizer: Adam (lr 1e-5, weight decay 1e-4)

## 4. Inference

```bash
python inference.py --model resnet18
python inference.py --model resnet50
```

By default this loads `output/<model>_weights.pt`; use `--ckpt <path>` to load another file. `--model` must match the architecture the weights were trained with.

## 5. Model comparison

Reads the per-epoch results from the training logs and plots the models together:

```bash
python plot_compare.py --logs logs/Res18_train_20261008_102219.log logs/train_20261008_150258.log --names ResNet18 ResNet50 --acc_ymin 75
```

Saves `output/compare_accuracy.png` (train / val accuracy) and `output/compare_f1_score.png` (val F1-score). These are Figures 2 and 3 in the report.

Prints the test accuracy, recall, precision and F1-score (positive class = PNEUMONIA), and saves `output/<model>_inference_confusion_matrix.png`.
