import os
import copy
import json
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet, DeepConvNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

PLOT_DIR = './plots'

def _plot_curves(curves, epochs, title, ylabel, filename):
    # curves: list of (values, label)
    os.makedirs(PLOT_DIR, exist_ok=True)
    x = range(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label in curves:
        ax.plot(x, values, label=label)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)

def plot_train_acc(train_acc_list, epochs):
    _plot_curves([(train_acc_list, 'Train')], epochs,
                 'Training Accuracy', 'Accuracy (%)', 'train_acc.png')

def plot_train_loss(train_loss_list, epochs):
    _plot_curves([(train_loss_list, 'Train')], epochs,
                 'Training Loss', 'Loss', 'train_loss.png')

def plot_test_acc(test_acc_list, epochs):
    best_epoch = int(np.argmax(test_acc_list)) + 1
    best_acc = max(test_acc_list)
    _plot_curves([(test_acc_list, 'Test')], epochs,
                 f'Testing Accuracy (best {best_acc:.2f}% @ epoch {best_epoch})',
                 'Accuracy (%)', 'test_acc.png')

def plot_test_loss(test_loss_list, epochs):
    _plot_curves([(test_loss_list, 'Test')], epochs,
                 'Testing Loss', 'Loss', 'test_loss.png')

def plot_train_test_curves(train_acc_list, test_acc_list, train_loss_list, test_loss_list, epochs):
    # Train vs. test comparison, as required by the report spec
    _plot_curves([(train_acc_list, 'Train'), (test_acc_list, 'Test')], epochs,
                 'Train / Test Accuracy', 'Accuracy (%)', 'train_test_acc.png')
    _plot_curves([(train_loss_list, 'Train'), (test_loss_list, 'Test')], epochs,
                 'Train / Test Loss', 'Loss', 'train_test_loss.png')

LOG_DIR = './logs'

def create_run_dir(args):
    # 每次 training 一個資料夾: logs/<時間>_<model>
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"
    run_dir = os.path.join(LOG_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def log_epoch(run_dir, epoch, train_loss, train_acc, test_loss, test_acc):
    # 每個 epoch 立即寫入，訓練中斷也保留紀錄
    path = os.path.join(run_dir, 'history.csv')
    row = pd.DataFrame([{'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                         'test_loss': test_loss, 'test_acc': test_acc}])
    row.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

def log_training(run_id, run_dir, args, model, train_acc_list, train_loss_list,
                 test_acc_list, test_loss_list):
    best_epoch = int(np.argmax(test_acc_list)) + 1
    record = {
        'run_id': run_id,
        'model': args.model,
        'activation': args.activation,
        'elu_alpha': args.elu_alpha,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.num_epochs,
        'weight_decay': args.weight_decay,
        'params': sum(p.numel() for p in model.parameters()),
        'best_test_acc': round(max(test_acc_list), 2),
        'best_epoch': best_epoch,
        'final_train_acc': round(train_acc_list[-1], 2),
        'final_test_acc': round(test_acc_list[-1], 2),
        'note': args.note,
    }

    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**record, 'architecture': str(model)}, f, indent=2, ensure_ascii=False)

    # 所有實驗彙整在同一張表，方便報告比較
    summary_path = os.path.join(LOG_DIR, 'summary.csv')
    pd.DataFrame([record]).to_csv(summary_path, mode='a',
                                  header=not os.path.exists(summary_path), index=False)
    print(f'Log saved to {run_dir}')

def train(model, loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    test_loss_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        test_acc, test_loss = test(model, test_loader, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = deepcopy(model.state_dict())
        print(f'Test Loss: {test_loss}')
        print(f'Test Acc. (%): {test_acc:3.2f}%')
        log_epoch(args.run_dir, epoch, avg_loss, avg_acc, test_loss, test_acc)

    print(f'Best Test Acc. (%): {best_acc:3.2f}%')
    torch.save(best_wts, os.path.join(args.run_dir, 'best.pt'))
    return avg_acc_list, avg_loss_list, test_acc_list, test_loss_list


def test(model, loader, criterion):
    avg_acc = 0.0
    avg_loss = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            avg_loss += criterion(outputs, labels.to(device)).item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100
        avg_loss /= len(loader.dataset)

    return avg_acc, avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-model", type=str, default='EEGNet', choices=['EEGNet', 'DeepConvNet'])
    parser.add_argument("-dropout", type=float, default=None,
                        help='default: 0.25 for EEGNet, 0.5 for DeepConvNet')
    parser.add_argument("-activation", type=str, default='ELU', choices=['ELU', 'ReLU', 'LeakyReLU'])
    parser.add_argument("-elu_alpha", type=float, default=1.0)
    parser.add_argument("-weight_decay", type=float, default=0.001)
    parser.add_argument("-note", type=str, default='', help='free-text note saved with the log')
    args = parser.parse_args()
    if args.dropout is None:
        args.dropout = 0.25 if args.model == 'EEGNet' else 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    run_id, args.run_dir = create_run_dir(args)
    PLOT_DIR = args.run_dir

    activation = {
        'ELU': nn.ELU(alpha=args.elu_alpha),
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
    }[args.activation]
    model_cls = EEGNet if args.model == 'EEGNet' else DeepConvNet
    model = model_cls(activation=activation, dropout=args.dropout)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.to(device)
    criterion.to(device)
    summary(model, (1, 2, 750), device=device.type)

    train_acc_list, train_loss_list, test_acc_list, test_loss_list = train(model, train_loader, criterion, optimizer, args)

    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)
    plot_test_loss(test_loss_list, args.num_epochs)
    plot_train_test_curves(train_acc_list, test_acc_list, train_loss_list, test_loss_list, args.num_epochs)
    log_training(run_id, args.run_dir, args, model, train_acc_list, train_loss_list,
                 test_acc_list, test_loss_list)
