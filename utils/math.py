import numpy as np

def RGBtoNORM(images):
    """Converting from RGB (0 to 255) to NORMalized value(-1 to 1)"""
    return (images.astype(float) / 127.5) - 1

def NORMtoRGB(images):
    """Converting from NORMalized value(-1 to 1) to RGB (0 to 255) """
    return ((images+1) * 127.5).astype(int)

def RGBtoHNORM(images):
    """Converting from RGB (0 to 255) to Half NORMalized value(0 to 1)"""
    return (images.astype(float) / 255)

def HNORMtoRGB(images):
    """Converting from Half NORMalized value(0 to 1) to RGB (0 to 255) """
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
    _min = np.min(images)
    _max = np.max(images)
    return ((images-_min)/(_max-_min) * 255).astype(int)
