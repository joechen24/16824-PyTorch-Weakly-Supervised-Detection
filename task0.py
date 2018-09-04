import visdom
import numpy as np
import _init_paths
from datasets.factory import get_imdb
import cv2

imdb = get_imdb('voc_2007_trainval')
print imdb._classes
print imdb._class_to_ind
pathto2018=imdb.image_path_at(2017)
print pathto2018
filename = pathto2018.split('/')[-1]
index = filename.split('.')[0]
annotation = imdb._load_pascal_annotation(index)
print annotation
gt_boxes = annotation['boxes']
gt_roidb = imdb.gt_roidb()
roidb = imdb._load_selective_search_roidb(gt_roidb)
roi2018 = roidb[2018]
##print roidb

vis = visdom.Visdom(port='8097')
myimg = cv2.imread(pathto2018)
for box in gt_boxes:
	bbox = tuple(box)
	cv2.rectangle(myimg, bbox[0:2], bbox[2:4], (0,204,0), 2)
for box in roi2018['boxes'][:10]:
	bbox = tuple(box)
	cv2.rectangle(myimg, bbox[0:2], bbox[2:4], (0,0,204), 2)
#print myimg
#cv2.imshow('origin',myimg)
myimg = myimg[:,:,-1::-1]
new_img = np.transpose(myimg, (2, 0, 1))
print new_img.shape
vis.image(new_img)
