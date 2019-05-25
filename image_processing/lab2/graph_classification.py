import numpy as np
import matplotlib.pyplot as plt
import networkx
import sknw

from skimage.io import imread
from skimage.filters import threshold_local
from skimage.morphology import (disk, remove_small_objects, dilation, erosion,
                                closing, opening, skeletonize, thin)
from skimage.measure import label
from skimage.segmentation import clear_border
from collections import Counter

def read(i, as_gray=False):
    img = imread('./data/' + str(i) + '.jpg', as_gray=as_gray)
    return img

def show(img, as_gray=True):
    if as_gray:
        plt.gray()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def get_figure(img, verbose=False):
    im = img < threshold_local(img, 401, offset=0.1)
    if verbose: show(im)
    im = remove_small_objects(im)
    if verbose: show(im)
    im = dilation(im, disk(5))
    if verbose: show(im)
    im = clear_border(im, buffer_size=1)
    if verbose: show(im)
    return im

def get_figure_colored(img, verbose=False):
    thresholds = [
        [[50, 90], [90, 140], [80, 130]],
        [[35, 65], [70, 100], [55, 95]],
        [[90, 140], [140, 200], [140, 190]]
    ]
    
    if verbose:
        plt.figure(figsize=(10,10))
        plt.suptitle('in get_figure_colored', fontsize=15)
        plt.gray()
        
    mask = np.zeros_like(img[:, :, 0], dtype=bool)
    for i, thr in enumerate(thresholds):
        and_mask = np.ones_like(img[:, :, 0], dtype=bool)
        for ch in range(3):
            cur_mask = (thr[ch][0] < img[:, :, ch]) & (img[:, :, ch] < thr[ch][1])
            and_mask &= cur_mask
            
        if verbose:
            plt.subplot(2,2,i+1)
            plt.axis('off')
            plt.title('threshold №{} mask'.format(i+1))
            plt.imshow(and_mask)
        
        mask |= and_mask
    
    if verbose:
        plt.subplot(2,2,4)
        plt.axis('off')
        plt.title('result mask')
        plt.imshow(mask)
        plt.show()
    
    img = remove_small_objects(mask, min_size=15)
    img = dilation(img, disk(10))
    img = closing(img, disk(10))
    labels = label(img)
    _, counts = np.unique(labels, return_counts=True)
    img = remove_small_objects(img, np.sort(counts)[-2])  
    
    for i in range(3):
        img = skeletonize(img)
        img = dilation(img, disk(10))
    
    if verbose:
        plt.title('figure')
        show(img)
        
    return img

def draw_graph(graph, img):
    plt.imshow(img, cmap='gray')
    
    for (s,e,n) in graph.edges:
        ps = graph[s][e][n]['pts']
        plt.plot(ps[:,1], ps[:,0], 'blue', lw=6)

    node, nodes = graph.node, graph.nodes()
    ps = np.array([node[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.', ms=20)
    for v in nodes: 
        plt.text(*node[v]['o'][::-1], f'{v}', fontsize=20, color='white')
    
    plt.axis('off')
#     plt.show()

def get_features(i, colored, verbose=0):
    img = read(i, as_gray=not colored)
    plt.title('picture №{}'.format(i))
    show(img, as_gray=False)
        
    fig = get_figure_colored(img, verbose==2) if colored else get_figure(img, verbose==2)
    if verbose == 1:
        plt.title('figure')
        show(fig)

    skel = skeletonize(fig)
    if verbose:
        plt.title('skeleton')
        show(dilation(skel, disk(5)))
    
    graph = sknw.build_sknw(skel, multi=True)

    if verbose == 2:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, verbose, 1)
        plt.title('unprocessed graph')
        draw_graph(graph, img)

    while True:
        for (u, v, n) in graph.edges:
            if len(graph[u][v][n]['pts']) < 47:
                graph.remove_edge(u, v, n)
                if u != v:
                    graph = networkx.contracted_nodes(graph, u, v)
                break
        else:
            break
            
    if verbose:
        plt.subplot(1, verbose, verbose)
        plt.title('graph')
        draw_graph(graph, img)
        plt.show()

    degrees = Counter(dict(graph.degree).values())
    degrees.pop(2, None)
    return degrees

def distance(f1, f2):
    return sum(((f1 - f2) + (f2 - f1)).values())

def classify(f):
    dists = list(map(lambda f2: distance(f, f2), [
        Counter({1: 3, 3: 3, 4: 3}),
        Counter({1: 4, 3: 4, 4: 1}),
        Counter({3: 5, 1: 4, 4: 2, 5: 1}),
        Counter({1: 6, 3: 4, 4: 2})
    ]))
    return np.argmin(dists) + 1