import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def histogram(tile, origin):
    hist = np.zeros(256)
    for i in range(tile.shape[0]):
        if tile[i] < origin:
            hist[tile[i]+2] += 1
        elif tile[i] == origin:
            hist[tile[i]+1] += 1
        else:
            hist[tile[i]] += 1
    return hist

def limitted_histogram(hist, clip_limit):
    add = 0
    for i in range(len(hist)):
        if hist[i] > clip_limit:
            add+= hist[i] - clip_limit
            hist[i] = clip_limit
    
    return hist+add/len(hist)

def cdf(hist):
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf

def clahe(img, tile_size, clip_limit):
    if tile_size % 2 != 0:
        add = 2
    else:
        add = 1

    border = tile_size // 2
    image = cv.copyMakeBorder(img, border, border, border, border, cv.BORDER_REFLECT)
    return_image = image.copy()
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            tile = image[i - border:i + border + add, j - border:j + border + add]
            tile = tile.flatten()
            if image[i,j] == 255:
                image[i,j] = 254
            tile_hist = histogram(tile, image[i,j])
            tile_hist = limitted_histogram(tile_hist, clip_limit)
            tile_cdf = cdf(tile_hist)
            tile_cdf = tile_cdf * 255 / tile_cdf[-1]
            return_image[i, j] = round(tile_cdf[image[i, j]])
            """cdf_norm = (tile_hist.cumsum() / (tile_size**2))*256
            return_image[i,j] = cdf_norm[image[i,j]]"""
    return return_image[border:-border, border:-border]
