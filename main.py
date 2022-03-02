#!/usr/local/bin/python3
# TURN IMAGINE INTO MOSAIC IMAGE:

# User selection - Provide jpg title of image you want to read in:
imgtit    = "parrot"      # without file ending (i.e. no ".jpg", ".png"...)
colnum    = 15               # how many different colours would you like your final image to have? Default 20 (can programmatically result in a few more if some colours are only present in very small amounts, see section 5)
contr_inc = "no"            # would you like to increase output img contrast (more vibrant)?
scale_fac = 10               # lanscape: 65. Animals: 10-30. Divide img. sizes (x, y) by this. More pixelated? -> higher value here.
seed      = 42               # random seed for k-means; not optimal colours output? Retry with different seed.

import cv2                   # install opencv-python to use
import copy
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.cluster import KMeans
from PIL import Image, ImageEnhance, ImageFont, ImageDraw


# 1.: PIXELATE IMAGE:
#####################
# Open img
img = Image.open(imgtit+".jpg")
# get img properties & scale factors:
orig_size = img.size
size      = tuple(axis / scale_fac for axis in orig_size) # box size (~ upscaled pixel). Dividing x AND y by same fac -> square blocks.
intsize   = [int(i) for i in size]

# Increase image contrast (potentially)
if contr_inc == "yes":
    enhancer = ImageEnhance.Contrast(img)
    img_out = enhancer.enhance(1.2)                         # by how much (>1 increases contrast)
else: # no contrast increase
    img_out = img

# Size down to n x n pixels
imgSmall = img_out.resize(intsize, resample=Image.BILINEAR)
# Scale back up using NEAREST to original size
pixelated = imgSmall.resize(img.size, Image.NEAREST)
pixelated.save(imgtit+"_pixelated.jpg")                   # save intermediary pixelated res (all colours)


# 2.: CLUSTER RGB COLOUR VALUES:
################################
# convert from PIL to array format:
pixelated_arr  = np.array(pixelated)  # r, g, b in correct order
# reshape the image for kmeans use (from grid to one after the other):
pixelated_resh = pixelated_arr.reshape((pixelated_arr.shape[1]*pixelated_arr.shape[0],3))
# applying kmeans:
kmeans = KMeans(n_clusters=colnum, random_state=seed)
fitted = kmeans.fit(pixelated_resh)
# get the labels (0 to colnum) the datapoints got assigned during clustering:
labels = list(kmeans.labels_)


# 3.: SELECT 1 COLOUR PER CLUSTER -> PROMINENT COLOURS:
#######################################################
# get the centroid info for each cluster (value in the middle):
centroids = kmeans.cluster_centers_


# 4.: FOR VERY SMALL CLUSTERS, SELECT > 1 COLOURS:
##################################################
# For the test image "blaumeise", the 2 vibrant blue tones appear in only very few boxes but are vital to the image.
# count how often each label (label = cluster belonging; from 0 to colnum-1) appears in the labels list = as a box in the pix. image:
perc_list = []; countlist_box = []
for numb in range(colnum):
    perc_list.append(int(labels.count(numb)/len(labels)*100))     # how large each cluster is [%]
    countlist_box.append(int(labels.count(numb) / scale_fac**2))  # Dividing by scale_fac * scale_fac brings countlist-results to the same dimensions as the reshaped pixelated (see clustering section)
                                                                  # => gives us BOX counts (pixelated) instead of pixels.
# get the indices (and thus label names) of those clusters appearing at a low %:
set_perc = 3  # from which % downwards small clusters are defined
perc_idx = [i for i,x in enumerate(perc_list) if x <= set_perc]
# get the indices of those labels in the full labels list, per label (= if label = 1, we want to know all indices where 1 appears)
labels_idx_list = []
for elem in perc_idx:
    labels_idx_list.append([i for i,x in enumerate(labels) if x == elem])

# check if the two additional colours are too similar to keep:
def checkDist(coll) -> object:
    r1, g1, b1 = coll[0][0], coll[0][1], coll[0][2]
    r2, g2, b2 = coll[1][0], coll[1][1], coll[1][2]
    dist = sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
    if dist <= 30 and dist != 0:
        """ dist can only = 0 when distance is to itself (bc 2 clusters will never 
        have the same centroid and additcols have been sorted for dist already) """
        # then keep only one of the two addit. colours for this cluster (too similar otherwise):
        coll = coll[0]
    return coll

# get the colours corresponding to the indices PER small colour cluster:
# for test img blaumeise those small clusters would be yellow, blue and white (3 out of 10).
def additionalColours(idx_list, all_colours):
    distrems = []
    col_coll = []
    breakps = [0.25, 0.75]        # index positions of taking 2 additional colours
    for cluster in idx_list:
        cols = [all_colours[idx] for idx in cluster]
        # select two colours per cluster:
        col_two = []              # per small cluster; two entries at most (one per breakpoint)
        for breakp in breakps:
            col = cols[int(len(cluster) * breakp)]
            col_two.append(col)
        # check if the two selected addit colours are too similar to one another:
        col_two = checkDist(col_two)
        if len(col_two) == 3:         # means there is just one element (containing r,g,b) inside
            distrem = "yes"
            col_coll.append(col_two)  # do not need to be separated by cluster any longer
        else:
            distrem = "no"
            col_coll = col_coll + col_two
        distrems.append(distrem)  # keep track if we have 1 or 2 addit cols per small cluster
    return col_coll, distrems     # distrem tells us that additional colours were removed due to too small distance

# apply additionalColours to small clusters:
addit_cols, distrems = additionalColours(labels_idx_list, pixelated_resh)
addit_cols = np.array(addit_cols)

# remove centroids for small clusters from centroids list:
centroids_clean = np.delete(centroids, perc_idx, 0)
# instead add two-colours-per-small-cluster colours:
addit_cols = addit_cols.astype('float64')
if addit_cols.size > 0:
    final_cols = np.concatenate([addit_cols, centroids_clean])
else:
    final_cols = centroids_clean

# for the final colours, check if any of them are too similar to one another as well:
def checkDist_allColours(coll) -> object:
    rememb   = []                   # keep track which colours were too close to one *another*
    remvd    = "no"                 # were any more colours removed in this process? (1 yes suffices)
    for colour in coll:
        othercolnum = 0
        for othercolour in coll:
            res = checkDist([colour, othercolour]) # both returned if dist big enough; 1 returned if not
            if len(res) == 1:
                remvd = "yes"
                rememb.append(coll[othercolnum])   # save deleted colour
                np.delete(coll, othercolnum)       # must be deleted right away or its similar colour will be deteled too in next it
            othercolnum += 1
    return np.array(coll), remvd, rememb

# keep original for later:
haystack = copy.deepcopy(final_cols)
# apply distance checker to all colours:
final_cols, remvd, rememb = checkDist_allColours(list(final_cols))


# 5.: REPRODUCE THE IMAGE WITH THE CENTROID COLOUR CLOSEST TO EACH BOX IMAGE:
#############################################################################
# crop/cut img to size divisible by scale_fac for simpler colour replacement below:
width, height = orig_size
#pixelated_cropped = pixelated.crop((0, 0, width-(width%scale_fac), height-(height%scale_fac)))

# selects out of dominant colours the one best fitting for original boxes:
def centroidMatcher(pixels):
    doms = []  # keeps track of dominant colours replacing the original ones
    shape = np.shape(pixels)
    width, height = shape[0], shape[1]      # in pixel which are still those of original image!
    for px in range(0, width, scale_fac):   # starts with 0
        for py in range(0, height, scale_fac):
            r, g, b = pixels[px, py]        # just look at one pixel per block -> uniform col
            # loop over choice list (most dominant colours) - check which is closest to current block:
            distlist = []
            for domcol in final_cols:
                domr, domg, domb = domcol
                dist = sqrt((domr - r)**2 + (domg - g)**2 + (domb - b)**2)
                distlist.append(dist)
            idx_min = np.argmin(distlist)
            rep_r, rep_g, rep_b = final_cols[idx_min]  # dominant replacement colour
            # set new colour to given box:
            pixels[px:px+scale_fac, py:py+scale_fac] = rep_r, rep_g, rep_b
            # add the dom colour selected to list:
            doms.append((rep_r, rep_g, rep_b))
    im = Image.fromarray(np.uint8(pixels))
    return im, doms

# reduce amount of colours in img and save:
simple_cols, doms = centroidMatcher(np.array(pixelated))
simple_cols.save(imgtit+"_simplified-colours.jpg")


# 6.: GET NUMBER OF DOMINANT COLOUR OCCURRENCES -> BUY / CUT FABRIC ACCORDINGLY:
################################################################################
# obtain box counts:
def DetermineBoxes(doms, doms_set) -> dict:
    box_counts = dict()  # will be filled with [colour]:[amount of boxes in that colour]
    for RGB in doms_set:
        box_counts[RGB] = doms.count(RGB)
    return box_counts
# count colour boxes in reduced-colour image:
boxes_counted = DetermineBoxes(doms, set(doms))

# output dominant RGBs + their counter:
for RGB in boxes_counted:
    print(RGB, ": ", boxes_counted[RGB], sep="")

# calculate new percentages of colours contained in simplified img:
domcols   = list(boxes_counted.keys())   # unique dominant final_cols
all       = sum(boxes_counted.values())  # sum of all blocks
perc_list = [el / sum(boxes_counted.values()) * 100 for el in list(boxes_counted.values())]


# 7.: MAKE NICE COLOUR PALETTE:
###############################
# sort the final colours by dist to white, so more similar shades appear close (potentially HSV cols here)
n = 0  # index for sorting
distlist = list()
for colour in domcols:
    r, g, b = colour
    dist = sqrt((r - 255) ** 2 + (g - 255) ** 2 + (b - 255) ** 2)
    distlist.append([dist, n])
    n += 1
# sort by distance; indices keep track of colour position in final_cols:
dist_sorted = sorted(distlist)
idx_list = [el[1] for el in dist_sorted]
final_doms = np.array([domcols[i] for i in idx_list])
# sort the percentages of the simplified colours likewise -> match domcol with correct percentage:
final_perc = np.array([perc_list[i] for i in idx_list]).round(1)

# select palette width, height based on amount of colours:
broader = 1    # factor, default 1 (=> square col tiles then)
tile    = 300  # how broad the single tiles of colours should be
palette_width  = len(final_doms) * tile
palette_height = tile * broader
# create array:
frame = np.zeros([palette_width, palette_height, 3], dtype=np.uint8)
# setting RGB start colours to white:
frame[:, :] = [255, 255, 255]

# fill with colours from my pixelRounder selection:
def paletteMaker(frame, summe=tile, n=0):
    for rgb_idx in range(len(final_doms)):  # as many times as there are colors in liste_unique
        while n < summe:                   # in increments of size tile; afterwards: jump to next colour
            idx = int(summe / tile - 1)                   # line below: frame[px,py]
            frame[n, 0:palette_height] = final_doms[idx]  # fills the height; index of 3-tuple containing a colour
            n += 1
        n = summe; summe += tile           # n=summe = n +1 once more.
    return(frame)

palette = paletteMaker(frame)
im = Image.fromarray(palette)


# 8.: HOW MUCH AREA DOES EACH FINAL DOMINANT COLOUR COVER? WRITE TO PALETTE:
############################################################################
# combine palette with final_perc in image:
def AddPercent(im):
    draw     = ImageDraw.Draw(im)
    myfont   = ImageFont.truetype("arial.ttf", 65)
    for perc_ind in range(len(final_perc)):
        fromleft = 20                     # indent on image from the left side
        fromtop  = 120 + tile*perc_ind    # indent from top; 120 for 1st box, 420 for 2nd...
        draw.text((fromleft, fromtop), str(final_perc[perc_ind])+"%", fill=(255, 255, 255),
              font=myfont, stroke_width=2, stroke_fill=(0, 0, 0))
    return im

# Add text to palette:
im_text = AddPercent(im)
im_text.save(imgtit+"_palette.jpg")