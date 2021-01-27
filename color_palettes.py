from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
