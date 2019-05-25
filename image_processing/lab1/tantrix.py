import numpy as np
import skimage

from skimage.io import imread
from skimage.filters import threshold_local, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import label, disk, remove_small_objects, convex_hull_object
from skimage.segmentation import clear_border
from skimage.transform import probabilistic_hough_line, rotate, resize
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist, rescale_intensity, equalize_adapthist

from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from scipy.ndimage import center_of_mass

import matplotlib.pyplot as plt


yellow = np.array([125, 150, 0])
red = np.array([150, 50, 0])
blue = np.array([100, 50, 100])
colors = [yellow, red, blue]
color_marks = ['y', 'r', 'b']
examples = None

example_cs = np.array(
  [['r', 'y', 'y', 'b', 'r', 'b'],
   ['r', 'r', 'b', 'y', 'y', 'b'],
   ['r', 'r', 'b', 'b', 'y', 'y'],
   ['b', 'r', 'y', 'b', 'y', 'r'],
   ['r', 'b', 'b', 'r', 'y', 'y'],
   ['b', 'y', 'b', 'r', 'y', 'r'],
   ['b', 'b', 'y', 'r', 'y', 'r'],
   ['b', 'b', 'r', 'y', 'r', 'y'],
   ['b', 'y', 'r', 'y', 'b', 'r'],
   ['r', 'b', 'y', 'y', 'r', 'b']]
)

def read_examples():
    examples = []
    for i in range(10):
        examples += [imread('./data/examples/{}.bmp'.format(i))]

def show_img(img, i=None):
    plt.gray()
    plt.imshow(img)
    plt.axis('off')
    #plt.savefig('./test_results/Single_{}.png'.format(i))
    #plt.savefig('./test_results/Group_{}.png'.format(i))
    plt.show()

def read_img_group(ind, as_gray=False):
    img = imread('./data/Group_' + str(ind) + '.bmp', as_gray=as_gray)
    return img

def read_img_single(ind, as_gray=False):
    img = imread('./data/Single_' + str(ind) + '.bmp', as_gray=as_gray)
    return img

def count_figs(img, return_masks=False, verbose=False):
    img = img < threshold_local(img, 601, offset=0.1)
    img = binary_fill_holes(img)
    img = clear_border(img, 2)
    img = convex_hull_object(img)
    labels, num = label(img, return_num=True)
    _, counts = np.unique(labels, return_counts=True)
    mean_area = np.mean(counts[counts != np.max(counts)])
    img = remove_small_objects(img, 0.6 * mean_area)
    labels, num = label(img, return_num=True)
    if verbose:
        show_img(img)
    if return_masks:
        masks = []
        unique_labels = np.unique(labels)
        for ul in unique_labels[1:]:
            masks += [labels == ul]
        return num, masks
    return num

def get_countour(mask):
    dist_map = distance_transform_edt(mask)
    contour = (dist_map >= 1) & (dist_map <= 4)
    return contour

def crop(img, mask):
    contour = get_countour(mask)
    
    lines = None
    for line_length in range(1, 101):
        lines = probabilistic_hough_line(contour, threshold=1, line_gap=4, line_length=line_length)
        if 6 <= len(lines) <= 14:
            break
    lines = np.array(lines)
    
    v = lines[0][1] - lines[0][0]
    cos = v.dot(np.array((1,0))) / np.sqrt(v.dot(v))
    angle = np.rad2deg(np.arccos(cos))

    r_min, c_min, r_max, c_max = regionprops(mask.astype(int))[0].bbox
    dr, dc = (r_max - r_min)//4, (c_max - c_min)//4  
    mask = mask[max(r_min-dr, 0):r_max+dr, max(c_min-dc, 0):c_max+dc]
    cropped_img = img[max(r_min-dr, 0):r_max+dr, max(c_min-dc, 0):c_max+dc, :]
    
    mask = rotate(mask, -angle)
    r_min, c_min, r_max, c_max = regionprops(mask.astype(int))[0].bbox
    cropped_img = (rotate(cropped_img, -angle) * mask[:, :, None])[r_min:r_max, c_min:c_max]
    cropped_img = resize(cropped_img, (100, 115))
    return cropped_img

def get_color_sequence(img):
    r, c, _ = img.shape
    color_sequence = []
    for i in range(6):
        im = skimage.img_as_ubyte(rotate(img, 60 * i))
        bit = np.array(im[r-20 : r-10, c//2 -8 : c//2 + 8])
        s = np.sum(bit, axis=2)
        bit_color = np.mean(bit[s > 50], axis=0)
        dist = []
        for color in colors:
            dist += [((color - bit_color) ** 2).sum()]
        color_sequence += [color_marks[np.argmin(dist)]]
    return np.array(color_sequence)

def distance(s1, s2):
    return (s1 == s2).sum()

def recognize_single(img, mask=None, show_res=False):
    if show_res:
        assert examples is not None
    if mask is None:
        _, masks = count_figs(rgb2gray(img), return_masks=True)
        mask = masks[0]
    im = crop(img, mask)
    p2, p98 = np.percentile(im, (2, 98))
    im = rescale_intensity(im, in_range=(p2, p98))
    im = rgb2hsv(im)
    im[:,:,2] = equalize_adapthist(im[:,:,2])
    im = hsv2rgb(im)
    color_sequence = get_color_sequence(im)
    variants_cs = []
    for i in range(len(color_sequence)):
        variants_cs += [np.roll(color_sequence, shift=i)]
    scores = np.zeros((len(variants_cs), len(example_cs)))
    for i, var in enumerate(variants_cs):
        for j, ex in enumerate(example_cs):
            scores[i, j] = distance(var, ex)
    scores = np.max(scores, axis=0)
    res = np.argmax(scores)
    if show_res:
        print('input image:')
        show_img(im)
        print('recognized as:')
        show_img(examples[res])
    return res + 1

def recognize_localize(img, verbose=True, i=-1):
    _, masks = count_figs(rgb2gray(img), return_masks=True)
    r = []
    coords = []
    for mask in masks:
        res = recognize_single(img, mask, show_res=verbose)
        r += [res]
        contour = get_countour(mask)
        img[contour, :] = np.array([255,0,0])
        coords += [center_of_mass(mask)]
        
    plt.figure
    for res, coord in zip(r, coords):
        plt.text(coord[1], coord[0], '{}'.format(res), fontsize=30, color='white')
    
    show_img(img, i)