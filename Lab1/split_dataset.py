import os
import random
import shutil
from argparse import ArgumentParser

CLASSES = ['NORMAL', 'PNEUMONIA']
SPLITS = ['train', 'val', 'test']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, default=os.path.join('raw', 'archive', 'chest_xray'))
    parser.add_argument('--dst', type=str, default='data')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f'{args.dst} already exists, remove it first to re-split')

    random.seed(args.seed)
    for c in CLASSES:
        # pool the original train/val/test images of this class
        files = []
        for s in SPLITS:
            folder = os.path.join(args.src, s, c)
            files += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        random.shuffle(files)

        # stratified split: same NORMAL:PNEUMONIA ratio in every split
        n_val = round(len(files) * args.val_ratio)
        n_test = round(len(files) * args.test_ratio)
        split_files = {'val': files[:n_val],
                       'test': files[n_val:n_val+n_test],
                       'train': files[n_val+n_test:]}

        for s, fs in split_files.items():
            out = os.path.join(args.dst, s, c)
            os.makedirs(out, exist_ok=True)
            for f in fs:
                # keep the original split in the name to avoid filename collisions
                orig_split = os.path.basename(os.path.dirname(os.path.dirname(f)))
                shutil.copy2(f, os.path.join(out, f'{orig_split}_{os.path.basename(f)}'))
            print(f'{s}/{c}: {len(fs)}')
