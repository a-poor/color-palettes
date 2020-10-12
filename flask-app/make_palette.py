
import io
import requests

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def img_from_url(url):
    return np.array(Image.open(io.BytesIO(requests.get(url).content)))

def img_from_file(path):
    with open(path,"rb") as f:
        return np.array(Image.open(f))
    
def flatten_and_scale(img):
    return img.reshape((-1,3)).astype("float32") / 255

def get_clusters(img,n_clusters=8):
    return KMeans(n_clusters).fit(img).cluster_centers_

def unscale(centers):
    return (centers * 255).astype("uint8").tolist()


def url_to_clusters(url,n_clusters=8):
    img = img_from_url(url)
    data = flatten_and_scale(img)
    centers = get_clusters(data,n_clusters)
    return unscale(centers)
