import os

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import io, color
from skimage.measure import label, regionprops
from skimage.morphology import dilation, closing, square

basedir = os.path.abspath(os.path.dirname(__file__))

INPUT_DIR = os.path.join(basedir, 'input')
OUTPUT_DIR = os.path.join(basedir, 'output')

"""
RGB-YCbCR
"""


def yuv_to_rgb(y, u, v):
    def clamp(n, smallest, largest): return max(smallest, min(n, largest))

    return [
        clamp(int(1.164 * (y - 16) + 1.596 * (v - 128)), 0, 255),
        clamp(int(1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128)), 0, 255),
        clamp(int(1.164 * (y - 16) + 2.018 * (u - 128)), 0, 255)
    ]


def rgb_to_yuv(r, r_max, g, g_max, b, b_max):
    return [
        (0.257 * r) + (0.504 * g) + (0.098 * b) + 16,
        (0.257 * r_max) + (0.504 * g_max) + (0.098 * b_max) + 16,
        -(0.148 * r) - (0.291 * g) + (0.439 * b) + 128,
        -(0.148 * r_max) - (0.291 * g_max) + (0.439 * b_max) + 128,
        (0.439 * r) - (0.368 * g) - (0.071 * b) + 128,
        (0.439 * r_max) - (0.368 * g_max) - (0.071 * b_max) + 128
    ]


# check the area of interest zone
def check_area(image, min_area, max_area):
    label_image = label(image)
    flag = False
    for region in regionprops(label_image):

        # take regions with large enough areas
        if min_area <= region.area <= max_area:
            flag = True
            break

    return flag


# search food
def find_food(image, y_min, y_max, u_min, u_max, v_min, v_max, name, min_area, max_area):
    print("Search: " + name)

    r_min, g_min, b_min = yuv_to_rgb(y_min, u_min, v_min)
    r_max, g_max, b_max = yuv_to_rgb(y_max, u_max, v_max)

    # color channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # mask for color channels
    mask_r = (r > r_min) & (r < r_max)
    mask_g = (g > g_min) & (g < g_max)
    mask_b = (b > b_min) & (b < b_max)

    # summary mask
    mask = mask_r & mask_g & mask_b

    # convert from rgb to gray
    mask1 = color.rgb2gray(mask)

    # fill holes
    mask1 = ndi.binary_fill_holes(mask1)

    # kernel for morphology
    s = 10
    selem = square(s)

    # dilation filter
    if not check_area(mask1, min_area, max_area):
        print("dilation " + str(s))
        mask2 = dilation(mask1, selem)
        mask2 = ndi.binary_fill_holes(mask2)
    else:
        mask2 = mask1

    # closing filter
    if not check_area(mask2, min_area, max_area):
        print("closing")
        mask3 = closing(mask2, selem)
        mask3 = ndi.binary_fill_holes(mask3)
    else:
        mask3 = mask2

    # segment an image with image labelling
    label_image = label(mask3)

    for region in regionprops(label_image):

        # take regions with large enough areas
        if min_area <= region.area <= max_area:
            # draw rectangle around segmented objects
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)

            # find center points in segmented objects
            y0, x0 = region.centroid

            # draw the name of food and its area
            ax.text(x0, y0, name + '\n' + str(region.area), fontsize=10, color='black')
            ax.set_title(u'Food detection')


if __name__ == '__main__':
    # open the input image in RGB Color space
    input_image_rgb = io.imread(os.path.join(INPUT_DIR, 'Меню (51).JPG'))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(input_image_rgb)

    # advanced
    find_food(input_image_rgb, 81.71, 140.675, 100.03, 88.315, 174.03, 178.0, 'Carrot', 290000, 400000)
    find_food(input_image_rgb, 72.8, 116.19, 112.5850, 114.805, 142.945, 136.36, 'Light bread', 180000, 250000)
    find_food(input_image_rgb, 102.935, 138.53, 100.155, 99.44, 142.88, 138.845, 'Soup', 200000, 800000)
    find_food(input_image_rgb, 44.122, 73.678, 116.752, 108.125, 161.364, 161.099, 'Tomato Juice', 96000, 120000)
    find_food(input_image_rgb, 56, 85, 110, 107, 152, 145, 'Cutlet', 50000, 100000)
    find_food(input_image_rgb, 140.938, 173.227, 94.586, 104.698, 145.856, 140.485, 'Puree', 180000, 270000)
    find_food(input_image_rgb, 37.98, 44.05, 122.77, 124.225, 155.825, 157.665, 'Juice 2', 40000, 200000)
    find_food(input_image_rgb, 21.742, 29.73, 125.188, 125.04, 136.341, 136.78, 'Juice 2', 40000, 200000)
    find_food(input_image_rgb, 84.435, 103.645, 93.495, 89.845, 162.635, 161.15, 'Juice 3', 40000, 200000)
    find_food(input_image_rgb, 34.164, 59.867, 119.017, 117.567, 143.565, 140.479, 'Apple juice', 50000, 100000)
    find_food(input_image_rgb, 102.081, 117.861, 109.073, 107.899, 143.513, 144.533, 'Bread', 40000, 150000)

    # set axis off
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    """
    Uncomment if u want to process multiple image
    """
    # i = 1
    # for img in io.imread_collection(os.path.join(INPUT_DIR, '*.JPG')):
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.imshow(img)
    #     ax.set_axis_off()
    #     plt.tight_layout()
    #
    #     find_food(input_image_rgb, 81.71, 140.675, 100.03, 88.315, 174.03, 178.0, 'Carrot', 290000, 400000)
    #     find_food(input_image_rgb, 72.8, 116.19, 112.5850, 114.805, 142.945, 136.36, 'Light bread', 180000, 250000)
    #     find_food(input_image_rgb, 102.935, 138.53, 100.155, 99.44, 142.88, 138.845, 'Soup', 200000, 800000)
    #     find_food(input_image_rgb, 44.122, 73.678, 116.752, 108.125, 161.364, 161.099, 'Tomato Juice', 96000, 120000)
    #     find_food(input_image_rgb, 56, 85, 110, 107, 152, 145, 'Cutlet', 50000, 100000)
    #     find_food(input_image_rgb, 140.938, 173.227, 94.586, 104.698, 145.856, 140.485, 'Puree', 180000, 270000)
    #     find_food(input_image_rgb, 37.98, 44.05, 122.77, 124.225, 155.825, 157.665, 'Juice 2', 40000, 200000)
    #     find_food(input_image_rgb, 21.742, 29.73, 125.188, 125.04, 136.341, 136.78, 'Juice 2', 40000, 200000)
    #     find_food(input_image_rgb, 84.435, 103.645, 93.495, 89.845, 162.635, 161.15, 'Juice 3', 40000, 200000)
    #     find_food(input_image_rgb, 34.164, 59.867, 119.017, 117.567, 143.565, 140.479, 'Apple juice', 50000, 100000)
    #     find_food(input_image_rgb, 102.081, 117.861, 109.073, 107.899, 143.513, 144.533, 'Bread', 40000, 150000)
    #
    #     plt.savefig(os.path.join(OUTPUT_DIR, str(i) + '-result.png'))
    #     i += 1
