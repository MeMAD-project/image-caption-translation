#! /usr/bin/env python3

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
import imageio
import argparse

parser = argparse.ArgumentParser(description='Image feature extraction for multimodal translation.')
parser.add_argument('images', metavar='FILES', type=str, nargs='*',
                    help='image filenames')
parser.add_argument('--imagelist', type=str, help='file containing image filenames')
parser.add_argument('--output', type=str, default='img_feat.npy',
                    help='output file to store features, if not "%(default)s"')
args = parser.parse_args()

images = []
if args.imagelist is not None:
    for i in open(args.imagelist).readlines():
        images.append(i.rstrip('\n'))
    
if len(args.images):
    images.extend(args.images)

if len(images)==0:
    print('No image filenames given, no output produced.')
    exit(1)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)

features = np.zeros((len(images), 80), dtype=np.float32)
                    
for i in range(len(images)):
    im = imageio.imread(images[i])
    outputs = predictor(im)
    #print(outputs['instances'].pred_classes)
    #print(outputs['instances'].pred_boxes)
    #print(outputs['instances'].pred_masks)

np.save(open(args.output, 'wb'), features)


