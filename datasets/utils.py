import numpy as np

def NormalizeImages(images):
    return (images.astype(float) / 127.5) - 1
