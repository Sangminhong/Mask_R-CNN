import sys
sys.path.insert(1, '/home/mchiash2/Mask_R-CNN/Mask_R-CNN')
import torch
import numpy as np 
#import dataset
#import utils.bbox
from anchor_gen import Anchor_Gen


class RPN:
## input images and then output the object porposal and objectness score
## assign positive label to two kind of anchors
## 1) highest IoU overlap with ground-truth box
## 2) IoU overlap higher than 0.7
## assign negative label when IoU with ground-truth box is lower than 0.3

    def __init__(self, img_w, img_h, sub_sample):

        a_gen = Anchor_gen(img_w, img_h, sub_sample)    
        anchors = a_gen.make_anchor()
        inside_index = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <=800)&
            (anchors[: ,3] <=800)
        ) [0]

        self.label = np.empty((len(inside_index)), )
        self.label.fill(-1)
        
        self.valid_anchor_boxes = anchors[inside_index]


    def calculate_Iou(self, bbox):
        ious = np.zeros((len(self.valid_anchor_boxes)),2)
        for num1, i in enumerate(self.valid_anchor_boxes):
            ya1, xa1, ya2, xa2 = i
            area_anchor = (ya2 - ya1)*(xa2 - xa1)
            for num2, j in enumerate(bbox):
                yb1, xb1, yb2, xb2 = j
                area_bbox = (yb2-yb1)*(xb2-xb1)

                inter_x1 = max(xb1,xa1)
                inter_y1 = max(yb1, ya1)
                inter_x2 = min(xb2, xa2)
                inter_y2 = min(yb2, ya2)

                if (inter_x1< inter_x2) and (inter_y1<inter_y2):
                    area_inter = (inter_y2 - inter_y1)*(inter_x2-inter_x1)
                    iou = area_iter/(anchor_area+box_area-inter_area)
                else:
                    iou = 0

                ious[num1, num2] = iou

        return ious



    def update_pos_label(self,ious):
        argmax_ious = ious.argmax(axio=0)
        get_max_ious = ious[gt_armgax_ious, np.arane(ious.shape[1])]


    def update_neg_label(self ious):    

