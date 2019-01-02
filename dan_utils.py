# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:54:07 2018

@author: shen1994
"""

import numpy as np
from scipy import ndimage

def get_meanshape():
    
    meanshape = np.array(
    [18.494846, 36.61314,  18.742294, 46.48093,  19.842281, 56.306053, 21.901043, 65.96931 , 
     25.723728, 74.954796, 31.63908 , 82.74449,  38.918457, 89.24516 , 46.927822, 94.50676 ,
     56.      , 96.05594 , 65.072174, 94.50676,  73.08154,  89.24516 , 80.360916, 82.74449 ,
     86.276276, 74.954796, 90.09895 , 65.96931,  92.157715, 56.306053, 93.257706, 46.48093 ,
     93.50516 , 36.61314 , 25.466898, 29.288471, 30.161482, 25.041758, 36.789574, 23.795765,
     43.62364 , 24.792725, 50.020443, 27.470646, 61.979557, 27.470646, 68.37636 , 24.792725,
     75.21043,  23.795765, 81.83852 , 25.041758, 86.533104, 29.288471, 56.      , 35.240334,
     56.      , 41.667118, 56.      , 48.046043, 56.      , 54.62276 , 48.454185, 58.960278,
     52.086994, 60.277733, 56.      , 61.446247, 59.913006, 60.277733, 63.545815, 58.960278,
     33.0997  , 35.995438, 37.131203, 33.620655, 42.01633,  33.695942, 46.26822 , 36.990746,
     41.678383, 37.84901 , 36.82142 , 37.774963, 65.73178 , 36.990746, 69.98367 , 33.695942,
     74.8688  , 33.620655, 78.9003,   35.995438, 75.17858 , 37.774963, 70.32162 , 37.84901 , 
     41.455666, 70.87096,  46.8126  , 68.76418 , 52.228645, 67.59149 , 56.      , 68.56457 , 
     59.771355, 67.59149 , 65.1874  , 68.76418 , 70.544334, 70.87096 , 65.35278 , 76.01848 ,
     60.10567 , 78.26798 , 56.      , 78.70176 , 51.89433 , 78.26798 , 46.647213, 76.01848 , 
     43.710133, 71.16683 , 52.17189 , 70.80757 , 56.      , 71.22449,  59.82811 , 70.80757 , 
     68.28986 , 71.16683 , 59.899345, 73.41396 , 56.,       73.87833 , 52.100655, 73.41396])
    
    return np.resize(meanshape, (68, 2))
    
def fit_to_rect(meanshape, box):

    box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ])

    box_w = box[2] - box[0]
    box_h = box[3] - box[1]

    meanshape_w = meanshape[:, 0].max() - meanshape[:, 0].min()
    meanshape_h = meanshape[:, 1].max() - meanshape[:, 1].min()

    w_s = box_w / meanshape_w
    h_s = box_h / meanshape_h
    scale = (w_s + h_s) / 2

    S0 = meanshape * scale

    S0_center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]
                
    S0 += box_center - S0_center

    return S0
    
def best_fit(destination, source, returnTransform=False):
    
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
    b = b / np.linalg.norm(srcVec)**2
    
    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean
    else:
        return np.dot(srcVec.reshape((-1, 2)), T) + destMean
    
def crop_resize_rotate(image, image_size, landmark, inputshape):
    
    A, t = best_fit(inputshape, landmark, True)
    A2 = np.linalg.inv(A)
    t2 = np.dot(-t, A2)

    out_image = ndimage.interpolation.affine_transform(image, \
                                                       A2, t2[[1, 0]], \
                                                       output_shape=[image_size, image_size])
    return out_image, [A, t]