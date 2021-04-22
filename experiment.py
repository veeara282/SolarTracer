import argparse
# TODO Implement our experimental design here

def parse_args():
    parser = argparse.ArgumentParser(description='Train and store the model')
    parser.add_argument('-o', '--out', metavar='model.pt', default='model.pt')
    parser.add_argument('-w', '--pos-class-weight', type=float, default=8.0)
    parser.add_argument('-e', '--num-epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=48)
    parser.add_argument('-m', '--mixed-precision', action='store_true')
    parser.add_argument('--train-dir', default='./SPI_train/')
    parser.add_argument('--val-dir', default='./SPI_val/')
    return parser.parse_args()

def main():
    pass

if __name__ == '__main__':
    main()
