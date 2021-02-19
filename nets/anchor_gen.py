import torch
import macpath
import numpy as np



class Anchor_Gen:

    def __init__(self, img_w, img_h, sub_sample):
        self.img_w = img_w
        self.img_h = img_h
        self.sub_sample = sub_sample
        self.ratios = [0.5, 1, 2]
        self.scales = [8, 16, 32]

        self.base_anchor = np.zeros((len(self.ratios)*len(self.scales), 4))



    def make_anchor(self):
        index = 0
        ctr=self.generate_ctr()
        anchors = np.zeros((self.fmap_size_w*self.fmap_size_h*9, 4))
        for c in ctr:
            ctr_y, ctr_x = c
            for i in range(len(self.ratios)):
                for j in range(len(self.scales)):
                    h = self.sub_sample*np.sqrt(self.ratios[i])*self.scales[j]
                    w = self.sub_sample*np.sqrt(1./self.ratios[i])*self.scales[j]

                        
                    anchors[index, 0] = ctr_y - h/2.
                    anchors[index, 1] = ctr_x - w/2.
                    anchors[index, 2] = ctr_y + h/2.
                    anchors[index, 3] = ctr_x + w/2.
                    index+=1


        return anchors

    def generate_ctr(self):
        a = self.sub_sample
        self.fmap_size_w = self.img_w//a
        self.fmap_size_h = self.img_h//a
        ctr_x = np.arange(a, (self.fmap_size_w+1)*a, a)
        ctr_y = np.arange(a, (self.fmap_size_h+1)*a, a)

        index=0
        ctr=np.zeros((len(ctr_x)*len(ctr_y),2))
        for i in range(len(ctr_x)):
            for j in range(len(ctr_y)):
                ctr[index, 1] = ctr_x[i] + a/2
                ctr[index, 0] = ctr_y[j] - a/2
                index+=1
                
        return ctr