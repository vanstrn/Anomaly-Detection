#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Mathematic functions for processing images and data.
"""
# ---------------------------------------------------------------------------

import numpy as np

def RGBtoNORM(images):
    """Converting from RGB (0 to 255) to NORMalized value(-1 to 1)"""
    return (images.astype(float) / 127.5) - 1

def NORMtoRGB(images):
    """Converting from NORMalized value(-1 to 1) to RGB (0 to 255) """
    images=Expand(images)
    return ((images+1) * 127.5).astype(int)

def RGBtoHNORM(images):
    """Converting from RGB (0 to 255) to Half NORMalized value(0 to 1)"""
    return (images.astype(float) / 255)

def HNORMtoRGB(images):
    """Converting from Half NORMalized value(0 to 1) to RGB (0 to 255) """
    images=Expand(images)
    return (images * 255).astype(int)

def UNWNtoNORM(images):
    """Converting from UNknoWN value(? to ?) to NORMalized value(-1 to 1)"""
    _min = np.min(images)
    _max = np.max(images)
    return ((images-_min)/(_max-_min)*2-1).astype(int)

def UNWNtoHNORM(images):
    """Converting from UNknoWN value(? to ?) to Half NORMalized value(0 to 1)"""
    _min = np.min(images)
    _max = np.max(images)
    return ((images-_min)/(_max-_min)).astype(int)

def UNWNtoRGB(images):
    """Converting from UNknoWN value(? to ?) to RGB (0 to 255) """
    images=Expand(images)
    _min = np.min(images)
    _max = np.max(images)
    return ((images-_min)/(_max-_min) * 255).astype(int)

def Expand(images):
    if np.shape(images)[-1]==1:
        return np.repeat(images,3,axis=-1)
    else:
        return images
