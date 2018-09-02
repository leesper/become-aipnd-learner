import argparse
import models
import json

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

model, class_to_idx = models.rebuild_model(args.checkpoint)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

for image in args.input:
    values, classes = models.predict(image, model, class_to_idx, args.gpu, args.top_k)
    flowers = [cat_to_name[cls] for cls in classes]
    for flower, prob in zip(flowers, values):
        print('{} ---> {}'.format(flower, prob))