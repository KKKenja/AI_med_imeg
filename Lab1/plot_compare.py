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
            # older logs have no training F1, this list is empty for them
            'train_f1': [float(x) for x in re.findall(r'Training F1-score: ([\d.]+)', text)],
            'val_acc': [float(x) for x in re.findall(r'Val Acc\.\(%\): ([\d.]+)', text)],
            'val_f1': [float(x) for x in re.findall(r'Precision: [\d.]+, F1-score: ([\d.]+)', text)]}


def plot_panel(ax, runs, key, title, ylabel, ymin=None):
    for i, (name, r) in enumerate(runs.items()):
        if r[key]:
            ax.plot(range(1, len(r[key])+1), r[key], color=f'C{i}', marker='o', ms=3, label=name)
    if ymin is not None:
        ax.set_ylim(ymin, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_phase(runs, phase, save_path, acc_ymin=None):
    # one figure per phase: accuracy on the left, F1-score on the right
    name = {'train': 'Training', 'val': 'Validation'}[phase]
    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_panel(ax_acc, runs, f'{phase}_acc', f'{name} Accuracy', 'Accuracy (%)', acc_ymin)
    plot_panel(ax_f1, runs, f'{phase}_f1', f'{name} F1-score', 'F1-score')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
        note = '' if r['train_f1'] else ' (no training F1 in this log)'
        print(f'{name}: {len(r["val_acc"])} epochs, best val acc {best:.2f}% (epoch {r["val_acc"].index(best)+1}){note}')

    os.makedirs(args.output_dir, exist_ok=True)
    plot_phase(runs, 'train', os.path.join(args.output_dir, 'compare_train.png'), args.acc_ymin)
    plot_phase(runs, 'val', os.path.join(args.output_dir, 'compare_val.png'), args.acc_ymin)
    print(f'## Comparison figures saved to {args.output_dir} (compare_train.png, compare_val.png) ##')
