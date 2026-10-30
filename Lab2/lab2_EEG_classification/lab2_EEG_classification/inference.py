import os
import json
import torch
import argparse
import dataloader
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.EEGNet import EEGNet, DeepConvNet


def build_model(cfg):
    activation = {
        'ELU': nn.ELU(alpha=cfg['elu_alpha']),
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
    }[cfg['activation']]
    model_cls = EEGNet if cfg['model'] == 'EEGNet' else DeepConvNet
    return model_cls(activation=activation, dropout=cfg['dropout'])


def evaluate(model, loader, device):
    correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            correct += (outputs.argmax(dim=1).cpu() == labels).sum().item()
    return correct, correct / len(loader.dataset) * 100


if __name__ == '__main__':
    # Load the saved best weights of one training run and evaluate them on the test set
    parser = argparse.ArgumentParser()
    parser.add_argument("-run", type=str, required=True, help='run folder, e.g. ./logs/20261029_171308_EEGNet')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(args.run, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)

    model = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(args.run, 'best.pt'), map_location=device, weights_only=True))
    model.to(device)

    _, _, test_data, test_label = dataloader.read_bci_data()
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_label, dtype=torch.int64))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct, acc = evaluate(model, test_loader, device)
    print('=' * 60)
    print(f"## Inference: {cfg['run_id']}")
    print(f"## Model: {cfg['model']}, activation: {cfg['activation']}, dropout: {cfg['dropout']}, "
          f"lr: {cfg['lr']}, batch size: {cfg['batch_size']}")
    print(f"## Weights: best.pt (epoch {cfg['best_epoch']})")
    print(f"## Correct: {correct} / {len(test_dataset)}")
    print(f"## Test Acc. (%): {acc:.2f}%")
    print('=' * 60)
