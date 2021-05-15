import argparse
from itertools import product
import os
import shutil

import numpy as np

'''Randomly assign files to training, validation, or test subsets.'''

rng = np.random.default_rng()

def parse_args():
    parser = argparse.ArgumentParser(description='Randomly assign files to training, validation, or test subsets')
    parser.add_argument('directory', metavar='path/to/directory', default='outputs/images')
    return parser.parse_args()

def create_subsets(root_dir):
    for subset, cls in product({'train', 'val', 'eval'}, {'0', '1'}):
        dir = os.path.join(root_dir, subset, cls)
        print(f'Create directory {dir}')
        os.makedirs(dir, exist_ok=True, mode=0o775)

def copy_to_random_subset(file, root_dir, cls):
    subset = rng.choice(['train', 'val', 'eval'], p=[0.7, 0.2, 0.1])
    dest = os.path.join(root_dir, subset, cls)
    print(f'Copy file {file} to {dest} (subset {subset})')
    shutil.copy(file, dest)

def randomize_class(root_dir, cls):
    path = os.path.join(root_dir, cls)
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                copy_to_random_subset(entry.path, root_dir, cls)

def main():
    args = parse_args()
    root_dir = args.directory
    # Create train/val/test directory structure
    create_subsets(root_dir)
    # Iterate through positive and negative classes
    randomize_class(root_dir, '0')
    randomize_class(root_dir, '1')

if __name__ == '__main__':
    main()
