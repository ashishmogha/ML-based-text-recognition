#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data,color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import os

image = imread(os.getcwd()+"/inputimage2.png")
image = color.rgb2gray(image)
# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

dummy = []

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        minr -= 2
        minc -= 2
        maxr += 2
        maxc += 2
        
        instance = bw[minr:maxr, minc:maxc]
        dummy.append(instance)
        #imshow(instance)
        
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()
