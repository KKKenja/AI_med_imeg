# Lab1 - Pneumonia Classification

Classifies chest X-ray images as **NORMAL** or **PNEUMONIA** with ImageNet-pretrained ResNet18 / ResNet50 (torchvision); the final `fc` layer is re-initialized for 2 classes.

Dataset: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Results

Final runs (batch size 32, seed 42, current code):

| Model | Best Val Acc (epoch) | Test Acc | Test F1 | Log |
|---|---|---|---|---|
| ResNet18 | 98.80% (28) | **97.26%** | **0.9813** | `logs/resnet18_train_20261015_134539.log` |
| ResNet50 | 98.63% (24) | 96.41% | 0.9751 | `logs/resnet50_train_20261015_183832.log` |

The test set is evaluated **only once**, using the checkpoint with the highest validation accuracy.
`output/resnet18_weights.pt` and `output/resnet50_weights.pt` are from these runs.
ResNet50 was trained twice with seed 42 (`logs/resnet50_train_20261015_143113.log` and `..._183832.log`) and the two logs are identical, so the results are reproducible.

Earlier runs (before the random seed was fixed and before training F1 was logged), used for the batch-size / run-to-run discussion in the report:

| Model | Batch size | Best Val Acc (epoch) | Test Acc | Test F1 | Log |
|---|---|---|---|---|---|
| ResNet18 | 64 | 98.46% (22) | 96.92% | 0.9787 | `logs/Res18_train_20261008_102219.log` |
| ResNet18 | 32 | 98.63% (23) | 96.92% | 0.9788 | `logs/resnet18_train_20261008_174911.log` |
| ResNet50 | 32 | 98.12% (29) | 98.12% | 0.9871 | `logs/train_20261008_150258.log` |

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
python train.py --model resnet18 --batch_size 32
python train.py --model resnet50 --batch_size 32
```

With the default `--seed 42`, rerunning these commands on our machine gave identical logs (results may still differ slightly on other GPUs / library versions).

Each run writes:

| Output | Description |
|---|---|
| `output/<model>_weights.pt` | Weights of the epoch with the highest val accuracy (change with `--ckpt`) |
| `output/<model>_accuracy.png` | Train / val accuracy curves |
| `output/<model>_f1_score.png` | Train / val F1-score curves |
| `output/<model>_confusion_matrix.png` | Confusion matrix on the test set |
| `logs/<model>_train_<time>.log` | Full training log |

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet18` | `resnet18`, `resnet50` or `resnet101` |
| `--num_epochs` | 30 | |
| `--batch_size` | 32 | Use 32 for ResNet50 on a 4 GB GPU |
| `--lr` | 1e-5 | Adam learning rate |
| `--wd` | 1e-4 | Weight decay |
| `--seed` | 42 | Random seed (Python / NumPy / PyTorch) |
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

Prints the test accuracy, recall, precision and F1-score (positive class = PNEUMONIA), and saves `output/<model>_inference_confusion_matrix.png`.

## 5. Model comparison

Reads the per-epoch results from the training logs and plots the runs together:

```bash
python plot_compare.py --logs logs/resnet18_train_20261015_134539.log logs/resnet50_train_20261015_183832.log --names ResNet18 ResNet50
```

Saves two figures, each with accuracy on the left and F1-score on the right:
- `output/compare_train.png`: training curves
- `output/compare_val.png`: validation curves

Logs from before training F1 was recorded (e.g. the batch-64 run) only show up in the accuracy panels.
