import numpy as np
import cv2
import scipy.ndimage

# Not cleaned yet. Check before run.

def setRandomWhiten(patch, max_light_num=3):
    light_canvas = np.zeros_like(patch)
    canvas = np.zeros_like(patch)
    light_num = np.random.randint(max_light_num+1)
    h, w, _ = patch.shape

    for i in range(h):
        val = i / float(h) 
        light_canvas[i,:] = val

    for i in range(light_num):
        tmp =  light_canvas * np.random.random()
        tmp = scipy.ndimage.interpolation.rotate(tmp, np.random.randint(360),
                reshape=False, order=0, mode='nearest')
        canvas = canvas + tmp

    canvas = np.maximum(0, np.minimum(0.3, canvas))
    output = np.maximum(0, np.minimum(1, canvas + patch))

    return output


def setRandomBackground(patch, max_light_num=3, scale=1):
    light_canvas = np.zeros_like(patch)
    canvas = np.zeros_like(patch)
    light_num = np.random.randint(max_light_num+1)
    h, w, _ = patch.shape

    for i in range(h):
        val = i / float(h) 
        light_canvas[i,:] = val

    for i in range(light_num):
        tmp =  light_canvas * np.random.random()
        tmp = scipy.ndimage.interpolation.rotate(tmp, np.random.randint(360),
                reshape=False, order=0, mode='nearest')
        canvas = canvas + tmp

    canvas = np.minimum(0.7, canvas)
    output = np.maximum(0, np.minimum(1, - canvas + patch))

    return output

def applyBlur(patch, ksize=3):
    h, w, _ = patch.shape
    patch = cv2.blur(patch, (ksize, ksize))
    return patch.reshape([h, w, 1])

def applyRandomBlur(patch):
    if np.random.random() < 0.7:
        return patch
    else:
        ksize = np.random.randint(1, 3)*2 + 1
        return applyBlur(patch, ksize)

def applyRandomNoise(patch):
    if np.random.random() < 0.7:
        return patch
    else:
        patch += np.random.normal(scale=0.1, size=patch.shape)
        patch = np.minimum(1.0, np.maximum(0.0, patch))
        return patch

def augmentPatch(patch):
    H, W, _ = patch.shape
    output = patch
    output = 1.0 - output # To think easier(black : 1.0, white : 0.0)
    output = applyRandomNoise(output)
    output = applyRandomBlur(output)
    if np.random.random() < 0.3:
        output = setRandomWhiten(output, 3)
    if np.random.random() < 0.5:
        output = setRandomBackground(output)
    output = 1.0 - output

    return output

def augmentLinewidth(patch):
    shape = patch.shape
    kernel = np.ones((3, 3), np.uint8)
    if np.random.random() < 0.5:
        output = patch
    else:
        num_iteration = np.random.randint(0, 3)
        output = cv2.erode(patch, kernel, iterations=num_iteration)
    output = output.reshape(shape)
    return output
