from pathlib import Path
from math import ceil
from functools import partial
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import color
from sklearn.cluster import KMeans, AgglomerativeClustering

from typing import Union, Callable


def read_image(path: Union[str,Path]) -> np.ndarray:
    with open(path,"rb") as f:
        return np.array(Image.open(f))

def get_kmeans_centers(img: np.ndarray, nclusters: int) -> np.ndarray:
    return KMeans(n_clusters=nclusters).fit(img).cluster_centers_

def get_agglom_centers(img: np.ndarray, nclusters: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=nclusters).fit(img).cluster_centers_
    
def rgb2hsv(dat: np.ndarray) -> np.ndarray:
    return color.rgb2hsv([dat])[0]

def hsv2rgb(dat: np.ndarray) -> np.ndarray:
    return color.hsv2rgb([dat])[0]

def preprocess_image(img: np.ndarray) -> np.ndarray:
    return img.reshape((-1,3)).astype("float32") / 255

def preprocess_rgb(img: np.ndarray) -> np.ndarray:
    return img.reshape((-1,3)).astype("float32") / 255

def preprocess_hsv(img: np.ndarray) -> np.ndarray:
    return rgb2hsv(preprocess_rgb(img))

def plot_image(img: np.ndarray):
    plt.figure(figsize=(14,8))
    plt.imshow(img)
    plt.grid()
    plt.axis('off')
    plt.show()

def plot_palette(centers: np.ndarray):
    plt.figure(figsize=(14,6))
    plt.imshow(centers[
        np.concatenate([[i] * 100 for i in range(len(centers))]).reshape((-1,10)).T
    ])
    plt.grid()
    plt.axis('off')
    plt.show()

def make_all_palettes(path: Union[str,Path], nclusters: int = 8, filter_fn: Callable[[np.ndarray],np.ndarray] = None):
    # Load the image
    img = read_image(path)
    
    # Reshape and set range
    rgb_pixels = preprocess_image(img)
    
    # If filter_fn is set, filter pixels
    if filter_fn is not None:
        rgb_pixels = filter_fn(rgb_pixels)
    
    # Convert RGB to HSV
    hsv_pixels = rgb2hsv(rgb_pixels)
    
    # Cluster the pixels
    km_rgb_centers = get_kmeans_centers(rgb_pixels,nclusters)
    km_hsv_centers = get_kmeans_centers(hsv_pixels,nclusters)
    km_hsv_centers = hsv2rgb(km_hsv_centers)
    
    agglom_rgb_centers = get_kmeans_centers(rgb_pixels,nclusters)
    agglom_hsv_centers = get_kmeans_centers(hsv_pixels,nclusters)
    agglom_hsv_centers = hsv2rgb(agglom_hsv_centers)
    
    # Plot the image
    plot_image(img)
    
    # Plot the palette
    print("KMeans RGB clustering")
    plot_palette(km_rgb_centers)
    print("KMeans HSV clustering")
    plot_palette(km_hsv_centers)
    
    print("Agglomerative RGB clustering")
    plot_palette(agglom_rgb_centers)
    print("Agglomerative HSV clustering")
    plot_palette(agglom_hsv_centers)
    
    print("\n" + "=" * 100 + "\n")

def filter_pixels(pixels: np.ndarray, low = 0.0, high = 1.0) -> np.ndarray:
    pix_mean = pixels.mean(1)
    mask = (low <= pix_mean) & (pix_mean <= high)
    idx = np.arange(len(pixels))
    return pixels[idx[mask]]


def palette_to_img(colors):
    """colors is a 2D array (palette-size, 3-channels)
    """
    assert len(colors.shape) == 2
    n_colors = len(colors)
    w = ceil(n_colors ** 0.75)
    h = ceil(n_colors / w)
    n_pad = (w * h) - n_colors
    if colors.max() <= 1: 
        c = colors * 255
    else:
        c = colors
    c = np.concatenate((c,255*np.ones((n_colors,1))),1)
    c = np.concatenate((c,np.zeros((n_pad,4),int)))
    c = c.reshape((h,w,4))
    img = Image.fromarray(c.astype("uint8"))
    return img.resize((w*200,h*200),Image.NEAREST)

def file_to_img(upload):
    with open(upload,"rb") as f:
        img = Image.open(f)
        return Image.fromarray(np.array(img))

def calc_palette(img: np.ndarray, hsv: bool = False, method: Union["kmeans","agglom"] = "kmeans", 
    nclusters: int = 5, filter_low: float = 0.0, filter_high: float = 1.0, callback: Callable = None):
    """
    """
    if callback: callback(0)
    cluster_fn = get_kmeans_centers if method.lower() == "kmeans" else get_agglom_centers

    if callback: callback(1)
    # Load the image
    # img = read_image(path)
    
    if callback: callback(2)
    # Reshape and set range
    pixels = preprocess_image(img)

    if callback: callback(3)
    # If filter_fn is set, filter pixels
    filter_fn = partial(filter_pixels,low=filter_low,high=filter_high) 
    pixels = filter_fn(pixels)

    if callback: callback(4)
    # Convert RGB to HSV
    if hsv: pixels = rgb2hsv(pixels)

    if callback: callback(5)
    # Cluster the pixels
    centers = cluster_fn(pixels,nclusters)
    if hsv: centers = hsv2rgb(centers)
    
    if callback: callback(9)
    return (centers * 255).astype(int)

def calc_palette_to_img(path: Union[str,Path], hsv: bool = False, method: Union["kmeans","agglom"] = "kmeans", 
    nclusters: int = 5, filter_low: float = 0.0, filter_high: float = 1.0, callback: Callable = None):
    """
    """
    assert isinstance(hsv,bool)
    assert method in ("kmeans","agglom")
    assert isinstance(nclusters,int)
    # assert 1 <= nclusters <= 35, nclusters
    assert 0 <= filter_low <= 1, filter_low
    assert 0 <= filter_high <= 1, filter_high
    assert filter_low < filter_high, str(filter_low,filter_high)

    palette = calc_palette(path,hsv,method,
        nclusters,filter_low,filter_high,callback)
    if callback: callback(10)
    return palette_to_img(palette)
    
