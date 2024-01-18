# encoding: utf-8

import math
from PIL import Image
import random
import  numpy as np
import random
import cv2
import torchvision.transforms as T
from defendAugment import *

########################### this code is for Local Grayscale Transformation(LGT = LGPR)  #################################
class LGPR(object): #Used to implement LTA ( Local Transformation Attack ) attack

    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        new = img.convert("L")
        np_img = np.array(new, dtype=np.uint8)
        img_gray = np.dstack([np_img, np_img, np_img])

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[1] and h < img.size[0]:
                x1 = random.randint(0, img.size[0] - h)
                y1 = random.randint(0, img.size[1] - w)
                img = np.asarray(img).astype('float')

                img[y1:y1 + h, x1:x1 + w, 0] = img_gray[y1:y1 + h, x1:x1 + w, 0]
                img[y1:y1 + h, x1:x1 + w, 1] = img_gray[y1:y1 + h, x1:x1 + w, 1]
                img[y1:y1 + h, x1:x1 + w, 2] = img_gray[y1:y1 + h, x1:x1 + w, 2]

                img = Image.fromarray(img.astype('uint8'))

                return img

        return img
#######################################################################################################################
################################ this code is for DL Defense  ################################################

def toSketch(img):  # Convert visible  image to sketch image
    img_np = np.asarray(img)
    img_inv = 255 - img_np
    img_blur = cv2.GaussianBlur(img_inv, ksize=(27, 27), sigmaX=0, sigmaY=0)
    img_blend = cv2.divide(img_np, 255 - img_blur, scale=256)
    img_blend = Image.fromarray(img_blend)
    return img_blend

"""
Randomly select several channels of visible image (R, G, B), gray image (gray), and sketch image (sketch) 
to fuse them into a new 3-channel image.
"""
def random_choose(r, g, b, gray_or_sketch):
    p = [r, g, b, gray_or_sketch, gray_or_sketch]
    idx = [0, 1, 2, 3, 4]
    random.shuffle(idx)
    return Image.merge('RGB', [p[idx[0]], p[idx[1]], p[idx[2]]])


#######################################################################################################
# Diversity Learning (DL = Fuse_LFusePR). G=grayscale, G_rgb=use Grayscale-RGB, S_rgb=fuse Sketch-RGB, Aug=more transformation, F=LHT
class Fuse_LFusePR(object):
    def __init__(self, G=0.05, G_rgb=0.01, S_rgb=0.01,Aug = 0.05,F = 0.1):
        self.G = G
        self.G_rgb = G_rgb
        self.S_rgb = S_rgb
        self.Aug = Aug
        self.F = F

    def __call__(self, img):
        r, g, b = img.split()
        gray = img.convert('L') #convert visible  image to grayscale images
        p = random.random()
        if p < self.G: #just Grayscale
            return Image.merge('RGB', [gray, gray, gray])

        elif p < self.G + self.G_rgb: #fuse Grayscale-RGB
            img2 = random_choose(r, g, b, gray)
            return img2

        elif p < self.G + self.G_rgb + self.S_rgb: #fuse Sketch-RGB
            sketch = toSketch(gray)
            img3 = random_choose(r, g, b, sketch)
            return img3
        elif p < self.G + self.G_rgb + self.S_rgb + self.Aug: #defendAugment
            policy = AutoAugPolicy()
            imgAug = policy(img)
            return imgAug
        elif p < self.G + self.G_rgb + self.S_rgb + + self.Aug + self.F: #DL with LHT(self.F means add LHT)
            img4 = fusePR(img)
            # pp = random.randint(0,300)
            # img4.save('./temp/'+str(pp)+'.jpg')
            return img4
        else:
            return img

def fusePR(img1):
    sl = 0.02
    sh = 0.4
    r1 = 0.3

    G = 0.2
    G_rgb = 0.2
    S_rgb = 0.2
    Aug = 0.4
    r, g, b = img1.split()
    gray = img1.convert('L') #convert visible  image to grayscale images
    p = random.random()
    img = img1
    if p < G: #just Grayscale
        img = Image.merge('RGB', [gray, gray, gray])

    elif p < G + G_rgb: #fuse Grayscale-RGB
        img = random_choose(r, g, b, gray)

    elif p < G + G_rgb + S_rgb: #fuse Sketch-RGB
        sketch = toSketch(gray)
        img = random_choose(r, g, b, sketch)
    elif p < G + G_rgb + S_rgb + Aug:
        policy = AutoAugPolicy()
        img = policy(img1)

    new = img

    np_img = np.array(new, dtype=np.uint8)

    for attempt in range(100):
        area = img1.size[0] * img1.size[1]
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img1.size[1] and h < img1.size[0]:
            x1 = random.randint(0, img1.size[0] - h)
            y1 = random.randint(0, img1.size[1] - w)
            img1 = np.asarray(img1).astype('float')

            img1[y1:y1 + h, x1:x1 + w, 0] = np_img[y1:y1 + h, x1:x1 + w, 0]
            img1[y1:y1 + h, x1:x1 + w, 1] = np_img[y1:y1 + h, x1:x1 + w, 1]
            img1[y1:y1 + h, x1:x1 + w, 2] = np_img[y1:y1 + h, x1:x1 + w, 2]

            img1 = Image.fromarray(img1.astype('uint8'))

            return img1
    return img1

