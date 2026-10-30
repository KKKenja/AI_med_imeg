# Lab2 - EEG Classification

Classifies 2-class motor-imagery EEG (BCI Competition III - IIIb, 2 bipolar channels) with **EEGNet** and **DeepConvNet** (bonus), both written by hand in PyTorch.

- Input: `[B, 1, 2, 750]` (2 channels x 750 time samples), output: `[B, 2]` logits
- Data: `S4b` + `X11b` subjects, 1080 train / 1080 test trials

## Results

<!-- TODO: fill in after the final runs; numbers come from logs/summary.csv -->

| Model | Activation | Dropout | LR | Best Test Acc (epoch) | Final-epoch Test Acc | Log |
|---|---|---|---|---|---|---|
| EEGNet | | | | | | `logs/<run_id>/` |
| DeepConvNet | | | | | | `logs/<run_id>/` |

> **Note:** "Best Test Acc" is the highest test accuracy over all epochs, and `best.pt` is the checkpoint from that epoch (the template's protocol). Because the epoch is picked by looking at the test set, this number is slightly optimistic. "Final-epoch Test Acc" is the unbiased one.

## Files

| Path | Description |
|---|---|
| `lab2_EEG_classification/lab2_EEG_classification/main.py` | Argument parsing, training/test loop, plotting, run logging |
| `lab2_EEG_classification/lab2_EEG_classification/models/EEGNet.py` | `EEGNet` and `DeepConvNet` |
| `lab2_EEG_classification/lab2_EEG_classification/dataloader.py` | Loads and concatenates the `.npz` files (provided by TA) |
| `lab2_EEG_classification/logs/` | One folder per training run + `summary.csv` |

## 1. Environment

```bash
pip install torch numpy pandas matplotlib tqdm torchsummary
```

## 2. Data

The dataset is not in the repo. Put the four files from E3 here:

```
lab2_EEG_classification/lab2_EEG_classification/data/
├── S4b_train.npz
├── S4b_test.npz
├── X11b_train.npz
└── X11b_test.npz
```

## 3. Training

`dataloader.py` uses paths relative to the **outer** `lab2_EEG_classification/` folder, so run from there:

```bash
cd Lab2/lab2_EEG_classification
python lab2_EEG_classification/main.py                                   # EEGNet, default settings
python lab2_EEG_classification/main.py -model DeepConvNet                # bonus model
python lab2_EEG_classification/main.py -activation ReLU -note "relu run" # experiment
```

| Argument | Default | Description |
|---|---|---|
| `-model` | `EEGNet` | `EEGNet` or `DeepConvNet` |
| `-num_epochs` | `150` | |
| `-batch_size` | `64` | |
| `-lr` | `0.001` | Adam learning rate |
| `-weight_decay` | `0.001` | Adam weight decay |
| `-activation` | `ELU` | `ELU`, `ReLU` or `LeakyReLU` |
| `-elu_alpha` | `1.0` | Alpha for ELU |
| `-dropout` | `0.25` (EEGNet) / `0.5` (DeepConvNet) | |
| `-note` | `""` | Free-text note saved with the run |

## 4. Outputs

Every run creates `logs/<YYYYmmdd_HHMMSS>_<model>/`:

| File | Content |
|---|---|
| `history.csv` | Train/test loss and accuracy per epoch (written every epoch) |
| `config.json` | All arguments, parameter count, best/final accuracy, model architecture |
| `best.pt` | Weights from the epoch with the highest test accuracy (not committed, see `.gitignore`) |
| `train_*.png`, `test_*.png`, `train_test_*.png` | Accuracy / loss curves |

`logs/summary.csv` gets one row per run, which is what the comparison tables in the report are built from.

## 5. Models

**EEGNet** (17,874 params): temporal Conv2d `(1,51)` → depthwise Conv2d `(2,1)`, groups=16 → separable Conv2d `(1,15)` → Linear(736, 2).

**DeepConvNet** (150,977 params): Conv2d `(1,5)` + Conv2d `(2,1)` → three blocks of Conv2d `(1,5)` with 50/100/200 filters, each block followed by BN, activation, MaxPool `(1,2)` and Dropout → Linear(8600, 2). Max-norm is omitted, which the spec allows.
