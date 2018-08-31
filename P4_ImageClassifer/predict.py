import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', 
    help='images of flowers')
parser.add_argument('checkpoint', nargs=1, 
    help='trained checkpoint')
parser.add_argument('--top_k', default=3, type=int, 
    help='top K classes')
parser.add_argument('--category_names', default='cat_to_name.json', 
    help='classes to real name')
parser.add_argument('--gpu', action='store_false', 
    help='predict in GPU model')

args = parser.parse_args()
print(args)