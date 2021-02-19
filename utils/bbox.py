import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class bbox:
    def __init__(self,y,x,width,height):
        
        self.bb = [y,x,width,height]

    def create_corner_rect(self, bb, color):
        bb = np.array(self.bb, dtype=np.float32)
        return plt.Rectangle((bb[1],bb[0]), bb[2],bb[3], color=color, fill=False, lw=3)

    def show_corner_bb(self,color):
        #plt.imshow(img)
        plt.gca().add_patch(self.create_corner_rect(self.bb, color=color))

          

    