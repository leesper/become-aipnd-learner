import argparse

parser = argparse.ArgumentParser(description='Train new network')
parser.add_argument('data_directory', nargs=1, required=True, help='a directory of training images')
parser.add_argument('--save_dir', nargs=1, help='a directory to save checkpoint')
parser.add_argument('--arch', nargs=1, help='network architecture')
parser.add_argument('--learning_rate', nargs=1, type=float, help='hyper parameter: learning rate')
parser.add_argument('--hidden_units', nargs=1, type=int, help='hypter parameter: hidden units')
parser.add_argument('--epochs', nargs=1, type=int, help='hyper parameter: epochs')
parser.add_argument('--gpu', action='store_false', help='train model in GPU mode')