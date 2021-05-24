from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from multi_resolution_segmentation import MultiResolutionSegmentation, train_multi_segmentation
from utils import train_transform, transform, to_device

import argparse
import numpy as np
import heapq
import json

# Use different learning rates
lr_round_1 = 1e-3
lr_round_2 = 3e-4

def random_search(train_loader: DataLoader,
                  val_loader: DataLoader,
                  ny_train_loader: DataLoader,
                  ny_val_loader: DataLoader,
                  num_trials: int = 10,
                  seed: int = 5670,
                  ny_num_epochs: int = 10):
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
        optimizer = optim.RMSprop(model.parameters(), alpha=0.9, momentum=0.9, lr=lr_round_1)
        print(f'Trial {t}: alpha = {alpha}, endpoints = {endpoints}')
        # Round 1
        metrics_round_1 = train_multi_segmentation(model, train_loader, val_loader, optimizer, num_epochs=1)
        # Round 2
        optimizer = optim.RMSprop(model.parameters(), alpha=0.9, momentum=0.9, lr=lr_round_2)
        metrics_round_2 = train_multi_segmentation(model, ny_train_loader, ny_val_loader, optimizer, num_epochs=ny_num_epochs)
        results.append({
            'trial': t,
            'alpha': alpha,
            'endpoints': endpoints,
            'model': model.cpu(), # Get it off the GPU to conserve memory
            'round_1': metrics_round_1,
            'round_2': metrics_round_2
        })
    return results

def fine_tuning(models_metadata: list, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
    for i, data in enumerate(models_metadata):
        model = to_device(data['model'])
        optimizer = optim.RMSprop(model.parameters(), alpha=0.9, momentum=0.9, lr=lr_round_2)
        metrics = train_multi_segmentation(model, train_loader, val_loader, optimizer, num_epochs=num_epochs)
        # These should add/update the fields of the item in models_metadata
        data['round_2'] = metrics
        data['model'] = model.cpu()
    return models_metadata

def print_results(results, round):
    for result in results:
        print(f"Trial {result['trial']}: alpha = {result['alpha']}, endpoints = {result['endpoints']}")
        print(f"Precision: {result[round]['precision']:.2%}")
        print(f"Recall: {result[round]['recall']:.2%}")
        print(f"F1: {result[round]['f1']:.2%}")
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
    parser.add_argument('--ny-train-dir', default='./NY_dataset/train/')
    parser.add_argument('--ny-val-dir', default='./NY_dataset/val/')
    return parser.parse_args()

def main():
    args = parse_args()

    # Use the original DeepSolar dataset for the first round of transfer learning
    train_set = ImageFolder(root=args.train_dir, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_set = ImageFolder(root=args.val_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Use our NY dataset for the second round
    ny_train_set = ImageFolder(root=args.ny_train_dir, transform=train_transform)
    ny_train_loader = DataLoader(ny_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    ny_val_set = ImageFolder(root=args.ny_val_dir, transform=transform)
    ny_val_loader = DataLoader(ny_val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    results = random_search(train_loader,
                            val_loader,
                            ny_train_loader,
                            ny_val_loader,
                            num_trials=args.num_trials,
                            seed=args.seed,
                            ny_num_epochs=10)
    
    # Print results for top k trials w.r.t. performance on NY dataset
    top_k = heapq.nlargest(3, results, key=lambda elt: elt['round_2']['f1'])
    print_results(top_k, 'round_2')

    # Save model with best performance on NY dataset
    best_model = top_k[0]['model']
    best_model.to_save_file(args.out)

    log_stats(results, args.logfile)

if __name__ == '__main__':
    main()
