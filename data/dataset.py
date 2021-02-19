import os
import torch
import numpy as np
import cv2  #OpenCV
import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import cocoeval
from pycocotools import mask
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, process, transform):
        """
            Args, variable: 
                root: image directory
                ann_path: coco annotation file path
                process: train/test/val
                transform: image transform

        """
        base_dir = os.path.abspath("/mnt/hdd2/COCO")
        if process is 'train':
            root=os.path.join(base_dir,"train2017")
            ann_path = os.path.join(base_dir, "annotations","instances_train2017.json")
        elif process is 'test':
            root=os.path.join(base_dir,"test2017")
            ann_path=os.path.join(base_dir, "annotations", "image_info_test2017.json")

        else:
            root=os.path.join(base_dir,"val2017")
            ann_path = os.path.join(base_dir, "annotations", "instances_val2017.json")

        images =[]
        images=self.load_images_from_folder(root)

        self.root = root 
        self.ann_path = ann_path       
        self.images=images
        self.coco = COCO(ann_path)   #laod dataset
        self.img_ids = self.coco.getImgIds()
        self.process = process
        self.transform = transform
        self.anns = self.coco.anns
        dataset = self.coco.dataset
        cats = self.coco.cats
        imgs = self.coco.imgs


        category_ids = []
        segmentations = []
        areas = []
        bboxs = []
        iscrowds =[] 
        index=0

        for ann in self.anns:
            category_ids.append(ann['category_id'])
            segmentations[index].append(ann['segmentation'])
            areas.append(ann['area'])
            bboxs[index].append(ann['bbox'])
            iscrowds.append(ann['iscrowd'])
            index+=1                          

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):    
        
        img_id = self.img_ids[index]
        origin_img = self.images[index]
        #origin_img = Img.fromarray(origin_img)
        new_img = self.trans(origin_img)
        ann_ids = self.coco.getAnnIds(imgIds= img_id)
        coco_anns = coco.loadAnns(ann_ids)
        
        return new_img, coco_anns



    def trans(self, img):
        trans = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            tr.Resize(800)

        ])

        return trans(img)

    
    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)

        return images        



def get_loader(process, transform, batch_size, shuffle, num_workers):

    coco=COCODataset(process = process, transform = transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)    

    return data_loader 

