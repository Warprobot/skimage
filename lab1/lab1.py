import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import io, color, feature, img_as_uint
from skimage.measure import label, regionprops
import os

basedir = os.path.abspath(os.path.dirname(__file__))

INPUT_DIR = os.path.join(basedir, 'input')
OUTPUT_DIR = os.path.join(basedir, 'output')


def find_row(dict, name, th):
    # sort the input seq
    sorted_y = sorted(dict.items(), key=lambda x: x[1])
    input = []
    for k, v in sorted_y:
        input.append(v)

    res = []
    for i in range(0, len(input) - 1):

        # if two points coord difference is less than the threshold
        if abs(input[i] - input[i + 1]) < th:
            if input[i] not in res:
                res.append(input[i])
            res.append(input[i + 1])
            if i == (len(input) - 2):
                if len(res) > 1:
                    print("Следующие точки по оси " + str(name) + " лежат примерно на одной прямой:")
                    for el in res:
                        for k in dict.keys():
                            if dict[k] == el:
                                print("[" + str(k) + "] ", el)
        else:
            if len(res) > 1:
                print("Следующие точки по оси " + str(name) + " лежат примерно на одной прямой:")
                for el in res:
                    for k in dict.keys():
                        if dict[k] == el:
                            print("[" + str(k) + "] ", el)
            res = []


if __name__ == '__main__':
    # input image as array of bytes
    input_image = io.imread(os.path.join(INPUT_DIR, 'phone.jpg'))

    # convert image from rgb to gray
    image_gray = color.rgb2gray(input_image)

    # save the gray copy of image into a file
    io.imsave(os.path.join(OUTPUT_DIR, 'rgb2gray.jpg'), image_gray)

    # find edges using Canny algorithm
    edges_canny = feature.canny(image_gray, sigma=2.9)

    # filled edges using mathematical morphology
    edges2 = ndi.binary_fill_holes(edges_canny)

    # saving iamge
    io.imsave(os.path.join(OUTPUT_DIR, 'canny.jpg'), img_as_uint(edges_canny))
    io.imsave(os.path.join(OUTPUT_DIR, 'detected.jpg'), img_as_uint(edges2))

    io.imshow(edges_canny)
    io.show()

    io.imshow(edges2)
    io.show()

    # segment an image with image labelling
    label_image = label(edges2)

    # create plot for result
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(edges_canny, cmap=plt.cm.gray)
    ax.set_title('Detected Objects')

    coord_y = {}
    coord_x = {}

    i = 1

    # find regions and center of each region after image labelling
    for region in regionprops(label_image):

        # take regions with large enough areas
        if region.area >= 150:
            # draw rectangle around segmented objects
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=2)

            ax.add_patch(rect)

            # find center points in segmented objects
            y0, x0 = region.centroid
            coord_y[i] = y0
            coord_x[i] = x0

            ax.plot(x0, y0, 'ws', markersize=6)
            ax.text(x0, y0, str(i), fontsize=12, color="red")

            print("[%s] x0: %s y0: %s" % (i, x0, y0,))
            i += 1

    # interface analysis
    find_row(coord_y, "y", 5)
    find_row(coord_x, "x", 5)

    # set axis off
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
