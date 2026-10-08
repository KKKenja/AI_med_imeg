import os
import sys
import copy
import warnings
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.models as models
from PIL import Image

import seaborn as sns
from matplotlib.ticker import MaxNLocator

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Logger:
    # write everything printed to both the terminal and a log file
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

class ChestXrayDataset(Dataset):
    # custom dataset: reads root/<split>/<class>/*.jpeg
    CLASSES = ['NORMAL', 'PNEUMONIA'] # label = index, PNEUMONIA (1) is the positive class
    EXTS = ('.jpeg', '.jpg', '.png')

    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.samples = []
        for label, cls in enumerate(self.CLASSES):
            folder = os.path.join(root, split, cls)
            # sorted for a deterministic order; skip non-image files such as .DS_Store
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(self.EXTS):
                    self.samples.append((os.path.join(folder, f), label))
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # X-rays are mostly grayscale (some are RGB); ImageNet-pretrained ResNets expect 3 channels
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(train, resize=224, degree=15):
    if train:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(p=0.25), # 25% 機率水平翻轉
            transforms.RandomRotation(degree),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)), # 小幅平移、縮放
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # 模擬不同曝光條件
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # 標準化
        ])
    # val / test: no augmentation, same resize and normalization as train
    return transforms.Compose([transforms.Resize((resize, resize)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def build_model(name, num_classes=2, pretrained=True):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f'Unknown model: {name}')
    # reinitialize the final classification layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def plot_accuracy(train_acc_list, val_acc_list, save_path):
    epochs = range(1, len(train_acc_list)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc_list, marker='o', label='Train Acc')
    plt.plot(epochs, val_acc_list, marker='o', label='Val Acc')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_f1_score(f1_score_list, save_path):
    epochs = range(1, len(f1_score_list)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, f1_score_list, marker='o', color='tab:green', label='Val F1-score')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Validation F1-score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(confusion_matrix, save_path):
    # test() returns [[tp, fn], [fp, tn]] (Pneumonia first);
    # reorder to [[tn, fp], [fn, tp]] so Normal comes first, same layout as the spec
    (tp, fn), (fp, tn) = confusion_matrix
    cm = np.array([[tn, fp],
                   [fn, tp]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['Actual Normal', 'Actual Pneumonia'])
    plt.title('Confusion Matrix (Test)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train(device, train_loader, val_loader, model, criterion, optimizer, num_epochs, ckpt_path):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []

    for epoch in range(1, num_epochs+1):

        with torch.set_grad_enabled(True):
            model.train()
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        # validation: only val is used to pick the best epoch, test is kept for the end
        val_acc, f1_score, _ = test(device, val_loader, model)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        f1_score_list.append(f1_score)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model_wts, ckpt_path)
    model.load_state_dict(best_model_wts)

    return train_acc_list, val_acc_list, f1_score_list

def test(device, test_loader, model, name='Val'):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]

        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ {name} Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # Windows terminal defaults to cp950, which cannot print '↳'
    sys.stdout.reconfigure(encoding='utf-8')

    parser = ArgumentParser()

    # for model
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50','resnet101'])
    parser.add_argument('--num_classes', type=int, required=False, default=2)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=1e-4)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='data') # re-split by split_dataset.py

    # for data augmentation
    parser.add_argument('--degree', type=int, default=15)
    parser.add_argument('--resize', type=int, default=224)

    # for output
    parser.add_argument('--output_dir', type=str, default='output') # weights and figures
    parser.add_argument('--ckpt', type=str, default=None) # best weights, default: <output_dir>/<model>_weights.pt

    args = parser.parse_args()

    # every run gets its own log file and outputs, prefixed by the model name
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join('logs', f'{args.model}_train_{datetime.now():%Y%m%d_%H%M%S}.log')
    sys.stdout = Logger(log_path)
    print(f'## Log file: {log_path} ##')
    print(f'## Args: {vars(args)} ##')

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader (custom ChestXrayDataset: NORMAL=0, PNEUMONIA=1)
    train_dataset = ChestXrayDataset(args.dataset, 'train', transform=get_transforms(True, args.resize, args.degree))
    val_dataset = ChestXrayDataset(args.dataset, 'val', transform=get_transforms(False, args.resize))
    test_dataset = ChestXrayDataset(args.dataset, 'test', transform=get_transforms(False, args.resize))
    print(f'## Dataset size - train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)} ##')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model (ImageNet pretrained, new 2-class fc layer)
    model = build_model(args.model, args.num_classes).to(device)

    # define loss function, optimizer
    # class weight = N / count_c, recomputed from the train split (same formula as the original 3.89 / 1.35)
    class_counts = np.bincount(train_dataset.targets)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(len(train_dataset) / class_counts))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training (best epoch is chosen by val acc)
    ckpt_path = args.ckpt or os.path.join(args.output_dir, f'{args.model}_weights.pt')
    train_acc_list, val_acc_list, f1_score_list = train(device, train_loader, val_loader, model, criterion, optimizer,
                                                        args.num_epochs, ckpt_path)
    print(f'## Best model saved to {ckpt_path} ##')

    # testing: evaluate the best model on the test set only once
    print('=' * 50)
    print('## Final Test ##')
    test_acc, test_f1_score, best_c_matrix = test(device, test_loader, model, name='Test')
    print('=' * 50)

    # plot
    plot_accuracy(train_acc_list, val_acc_list, os.path.join(args.output_dir, f'{args.model}_accuracy.png'))
    plot_f1_score(f1_score_list, os.path.join(args.output_dir, f'{args.model}_f1_score.png'))
    plot_confusion_matrix(best_c_matrix, os.path.join(args.output_dir, f'{args.model}_confusion_matrix.png'))
