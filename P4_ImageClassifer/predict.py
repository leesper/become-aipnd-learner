import argparse
import common
import json

parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='+',
    help='images of flowers')
parser.add_argument('checkpoint', nargs=1, 
    help='trained checkpoint')
parser.add_argument('--top_k', default=[5], type=int, 
    help='top K classes')
parser.add_argument('--category_names', default=['cat_to_name.json'], 
    help='classes to real name')
parser.add_argument('--gpu', action='store_true', 
    help='predict in GPU model')

args = parser.parse_args()

images = args.images
checkpoint = args.checkpoint[0]
top_k = args.top_k[0]
category_names = args.category_names[0]
gpu = args.gpu

print('checkpoint: {}'.format(checkpoint))
print('TOP K: {}'.format(top_k))
print('category names: {}'.format(category_names))
print('training on GPU: {}'.format(gpu))

model, class_to_idx = common.rebuild_model(checkpoint)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

for image in images:
    values, classes = common.predict(image, model, class_to_idx, gpu, top_k)
    flowers = [cat_to_name[cls] for cls in classes]
    print('image {}'.format(image))
    for flower, prob in zip(flowers, values):
        print('{} ---> {}'.format(flower, prob))
    print('-' * 10)