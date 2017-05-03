import os

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color, draw
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, closing, square

basedir = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(basedir, 'output')

# Haar primitives
haar = {
    'haar1':
        np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),

    'haar2':
        np.array([[1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),

    'haar3':
        np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),

    'haar4':
        np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]]),

}


# show primitive
# io.imshow(haar['haar12'])
# io.show()

# primitives searching
def primitive_search(img, haar_type, th1, th2, point, marker_size, draw_haar):
    # input image rows and cols calculating
    rows = img.shape[0]
    cols = img.shape[1]

    prim_amount = 0
    cur_haar = haar[haar_type]  # init haar primitive

    print("%s - %s" % (haar_type, point))

    res = []
    arr = []

    # for each pixel in image with step of size window
    for row in range(0, rows, win_size):
        for col in range(0, cols - 9, 2):
            b = 0
            w = 0

            # for each pixel in window
            for px in range(0, win_size):
                for py in range(0, win_size):

                    # if pixel is under '1' in haar primitive
                    if cur_haar[py][px] == 1:
                        b += img[row + py][col + px]
                    # if pixel is under '0' in haar primitive
                    else:
                        w += img[row + py][col + px]

            if th1 < abs(b - w) < th2:
                prim_amount += 1

                # add marker in window
                if not draw_haar:
                    ax.plot(col + win_size / 2, row + win_size / 2, point, markersize=marker_size)
                    arr.append([col + win_size / 2, row + win_size / 2])


                # draw haar primitive
                else:
                    for px in range(0, win_size):
                        for py in range(0, win_size):
                            if cur_haar[py][px] == 1:
                                ax.plot(col + px, row + py, 'bs', markersize=1)

                            else:
                                ax.plot(col + px, row + py, 'ws', markersize=1)

                # ax.text(col + win_size / 2, row + win_size / 2, str(abs(b - w))[:5], fontsize=10, color="red")

            res.append(b - w)

    print("max: %s" % max(res))
    print("min: %s" % min(res))
    print("amount: %s" % prim_amount)

    return arr

    # fig.savefig(haar_type + '.png')


def draw_pix(img, coord, amount, r, g, b):
    rows = img.shape[0]
    cols = img.shape[1]

    for el in bus:
        col, row = el
        if bus.count(el) == amount:
            rr, cc = draw.circle(row, col, 7, img.shape)
            img[rr, cc, :] = (r, g, b)

    for row in range(0, rows):
        for col in range(0, cols):
            if [col, row] in coord:
                if img[row, col, 0] != r and img[row, col, 1] != g and img[row, col, 2] != b:
                    # draw circle in interesting point
                    img[row, col, :] = (r, g, b)
                    rr, cc = draw.circle(row, col, 7, img.shape)
                    img[rr, cc, :] = (r, g, b)

    return img

# open the input image
image1 = io.imread(os.getcwd() + '/pool/018.jpg')

# convert image from rgb to gray
image2 = color.rgb2gray(image1)

win_size = 10

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(image1)

# array for interesting points
bus = []
result = []

bus += primitive_search(image2, 'haar1', 34.2, 34.5, 'ws', 4, False)
bus += primitive_search(image2, 'haar2', 24, 25, 'y^', 2, False)
bus += primitive_search(image2, 'haar3', 31, 33, 'ro', 2, False)
bus += primitive_search(image2, 'haar4', 10, 15, 'ys', 4, False)

# draw interesting points
image3 = draw_pix(image1, bus, 2, 255, 255, 255)

# help image for result
image4 = color.rgb2gray(image3)
io.imsave(os.path.join(OUTPUT_DIR, 'rgb2gray.jpg'), image4)

selem = square(15)

# dilation morph
image5 = dilation(image4, selem)
io.imsave(os.path.join(OUTPUT_DIR, 'dilation.jpg'), image5)

# closing morph
image6 = closing(image5, selem)
io.imsave(os.path.join(OUTPUT_DIR, 'closing.jpg'), image6)

# segment an image with image labelling
label_image = label(image6)

# for each interesting region
for region in regionprops(label_image):

    # take regions with large enough areas
    if 500 < region.area < 800:
        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='blue', linewidth=1)

        # add rect to interesting area
        ax.add_patch(rect)
        y0, x0 = region.centroid
        ax.text(x0, y0, str(region.area), fontsize=15, color='blue')

# set axis off
ax.set_axis_off()
fig.savefig(os.path.join(OUTPUT_DIR, 'result.png'))
plt.show()
