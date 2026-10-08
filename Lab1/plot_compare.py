import re
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_log(path):
    # only read the per-epoch part, so the final test result is not mixed in
    text = open(path, encoding='utf-8').read().split('## Final Test ##')[0]
    return {'train_acc': [float(x) for x in re.findall(r'Training Acc\.\(%\): ([\d.]+)', text)],
            'val_acc': [float(x) for x in re.findall(r'Val Acc\.\(%\): ([\d.]+)', text)],
            'val_f1': [float(x) for x in re.findall(r'F1-score: ([\d.]+)', text)]}


def plot_accuracy(runs, save_path, ymin=None):
    plt.figure(figsize=(8, 5))
    for i, (name, r) in enumerate(runs.items()):
        epochs = range(1, len(r['train_acc'])+1)
        plt.plot(epochs, r['train_acc'], '-', color=f'C{i}', marker='o', ms=3, label=f'{name} Train')
        plt.plot(epochs, r['val_acc'], '--', color=f'C{i}', marker='s', ms=3, label=f'{name} Val')
    if ymin is not None:
        plt.ylim(ymin, 100)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy: ' + ' vs '.join(runs))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_f1_score(runs, save_path):
    plt.figure(figsize=(8, 5))
    for i, (name, r) in enumerate(runs.items()):
        plt.plot(range(1, len(r['val_f1'])+1), r['val_f1'], '-', color=f'C{i}', marker='o', ms=3, label=f'{name} Val F1')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Validation F1-score: ' + ' vs '.join(runs))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True, help='training logs written by train.py')
    parser.add_argument('--names', nargs='+', required=True, help='one name per log, e.g. ResNet18 ResNet50')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--acc_ymin', type=float, default=None, help='lower limit of the accuracy y-axis')
    args = parser.parse_args()
    assert len(args.logs) == len(args.names), '--logs and --names must have the same length'

    runs = {name: read_log(path) for name, path in zip(args.names, args.logs)}
    for name, r in runs.items():
        best = max(r['val_acc'])
        print(f'{name}: {len(r["val_acc"])} epochs, best val acc {best:.2f}% (epoch {r["val_acc"].index(best)+1})')

    os.makedirs(args.output_dir, exist_ok=True)
    plot_accuracy(runs, os.path.join(args.output_dir, 'compare_accuracy.png'), args.acc_ymin)
    plot_f1_score(runs, os.path.join(args.output_dir, 'compare_f1_score.png'))
    print(f'## Comparison figures saved to {args.output_dir} ##')
