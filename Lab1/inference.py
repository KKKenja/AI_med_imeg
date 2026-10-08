import os
import sys
import warnings
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from train import ChestXrayDataset, build_model, get_transforms, test, plot_confusion_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    # Windows terminal defaults to cp950, which cannot print '↳'
    sys.stdout.reconfigure(encoding='utf-8')

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument('--ckpt', type=str, default=None, help='weights saved by train.py, default: <output_dir>/<model>_weights.pt')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = os.path.join(args.output_dir, f'{args.model}_weights.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    test_dataset = ChestXrayDataset(args.dataset, 'test', transform=get_transforms(False, args.resize))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # no need to download ImageNet weights, they are overwritten by the checkpoint
    model = build_model(args.model, args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)

    print('=' * 50)
    print(f'## Inference: {args.model} ({args.ckpt}) ##')
    test_acc, test_f1_score, c_matrix = test(device, test_loader, model, name='Test')
    print('=' * 50)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.model}_inference_confusion_matrix.png')
    plot_confusion_matrix(c_matrix, save_path)
    print(f'## Confusion matrix saved to {save_path} ##')
