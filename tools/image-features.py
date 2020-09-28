#! /usr/bin/env python3

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import numpy as np
import imageio
import argparse

# model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model_name = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
score_threshold = 0.0

parser = argparse.ArgumentParser(description='Image feature extraction for multimodal translation.')
parser.add_argument('images', metavar='FILES', type=str, nargs='*',
                    help='image filenames')
parser.add_argument('--cpu', action='store_true', help='force to use CPU even if CUDA is available')
parser.add_argument('--imglist', type=str, help='file containing image filenames')
parser.add_argument('--output', type=str, default='img_feat.npy',
                    help='output file to store features, if not "%(default)s"')
args = parser.parse_args()

images = []
if args.imglist is not None:
    for i in open(args.imglist).readlines():
        images.append(i.rstrip('\n'))

if len(args.images):
    images.extend(args.images)

if len(images) == 0:
    print('No image filenames given, no output produced.')
    exit(1)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

print('Extracting detectron2 mask surface features for',
      len(images), 'images with', end=' ')

if args.cpu or not torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cpu'
    print('CPU')
else:
    print('CUDA')

predictor = DefaultPredictor(cfg)

features = np.zeros((len(images), 80), dtype=np.float32)

for i in range(len(images)):
    im = imageio.imread(images[i])
    outputs = predictor(im)
    pred_masks = outputs["instances"].pred_masks.cpu().numpy()
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()

    for j, c in enumerate(pred_classes):
        img_size = pred_masks[j].shape
        tot_count = img_size[0] * img_size[1]
        ms = np.count_nonzero(pred_masks[j]) / tot_count
        features[i, c] += ms  # mask_surface
        # features[i, c] += 1   # bbox_count

np.save(open(args.output, 'wb'), features)

print('Stored features in "', args.output, '"', sep='')
