from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from segmentation import to_device, transform
from multi_resolution_segmentation import MultiResolutionSegmentation, train_multi_segmentation

import argparse
import numpy as np
from operator import itemgetter
import heapq
import json
# TODO Implement our experimental design here

def random_search(train_loader: DataLoader, val_loader: DataLoader, num_trials: int = 10, seed: int = 5670):
    rng = np.random.default_rng(seed)
    alpha_values = rng.uniform(4, 16, num_trials)
    # To generate combinations of endpoints:
    # 1. For each trial, generate 4 random integers in {0, 1}
    # 2. If all of them are 0, generate another 4 random integers
    # 3. endpoints_values[t, k - 1] means that endpoint k should be used in trial t
    endpoints_values = rng.integers(0, 2, (num_trials, 4))
    for t in range(num_trials):
        # Keep generating new random integers until at least one of them is 1
        while not np.any(endpoints_values[t]):
            endpoints_values[t] = rng.integers(0, 2, 4)
    # Store results and print them out at the end
    results = []
    # Start random search
    for t in range(num_trials):
        alpha = alpha_values[t]
        # Class constructor expects a list of endpoint numbers, not a bit vector
        endpoints = [k + 1 for k in range(4) if endpoints_values[t, k]]
        # Create and train model
        model = to_device(MultiResolutionSegmentation(pos_class_weight=alpha, endpoints=endpoints))
        # Use RMSProp parameters from the DeepSolar paper (alpha = second moment discount rate)
        # except for learning rate decay and epsilon
        optimizer = optim.RMSprop(model.parameters(), alpha=0.9, momentum=0.9, eps=0.001, lr=1e-3)
        print(f'Trial {t}: alpha = {alpha}, endpoints = {endpoints}')
        metrics = train_multi_segmentation(model, train_loader, val_loader, optimizer, num_epochs=1)
        results.append({
            'trial': t,
            'alpha': alpha,
            'endpoints': endpoints,
            **metrics,
            'model': model.cpu() # Get it off the GPU to conserve memory
        })
    return results

def print_results(results):
    for result in results:
        print(f"Trial {result['trial']}: alpha = {result['alpha']}, endpoints = {result['endpoints']}")
        print(f"Precision: {result['precision']:.2%}")
        print(f"Recall: {result['recall']:.2%}")
        print(f"F1: {result['f1']:.2%}")
        print()

def log_stats(results, log_file):
    # Make a copy of the results dicts minus the model key
    results_without_models = [{k: v for k, v in res.items() if k != 'model'} for res in results]
    with open(log_file, 'w') as f:
        json.dump(results_without_models, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and store the model')
    parser.add_argument('-o', '--out', metavar='model.pt', default='random_search.pt')
    parser.add_argument('--logfile', metavar='log_file.json', default='random_search.json')
    parser.add_argument('-n', '--num-trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=5670)
    parser.add_argument('-b', '--batch-size', type=int, default=48)
    # parser.add_argument('-m', '--mixed-precision', action='store_true')
    parser.add_argument('--train-dir', default='./SPI_train/')
    parser.add_argument('--val-dir', default='./SPI_val/')
    return parser.parse_args()

def main():
    args = parse_args()

    train_set = ImageFolder(root=args.train_dir, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_set = ImageFolder(root=args.val_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    results = random_search(train_loader, val_loader, num_trials=args.num_trials, seed=args.seed)
    top_k = heapq.nlargest(3, results, key=itemgetter('f1'))
    print_results(top_k)

    best_model = top_k[0]['model']
    best_model.to_save_file(args.out)

    log_stats(results, args.logfile)

if __name__ == '__main__':
    main()
