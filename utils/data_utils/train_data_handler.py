from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import cv2
import os
import sys

import numpy as np
import scipy
import scipy.misc
import scipy.ndimage

class TrainDataHandler(object):
    """Abstracted data Handler that contains common functions 
    for train data handler.
    """
    def __init__(self, batch_size, target_size, is_color, num_threads):
        self.batch_size = batch_size
        self.target_size = target_size
        self.is_color = is_color
        if self.is_color:
            self.CH = 3
        else:
            self.CH = 1

        self.num_threads = num_threads
        self._queue = Queue(100)
        self.msg_queue = Queue(1)
        self.procs = []

    @abstractmethod
    def _enqueue_op(self):
        """Generate a batch of train data and enqueue to the global queue.
        Each thread calls this function and enqueue data asynchronously.
        Data handler that inherits this class must implement this 
        _enqueue_op function.
        """
        pass

    def startThreads(self):
        """Start multiple threads that call _enqueue_op function.
        """
        for i in range(self.num_threads):
            proc = Process(target=self._enqueue_op, 
                    args=(self._queue, self.msg_queue, i))
            self.procs.append(proc)
            proc.daemon = True
            proc.start()

    def kill(self):
        """Kill all threads for clean halt.
        """
        for proc in self.procs:
            proc.terminate()
            proc.join()

    def next(self):
        """Provide next batches.

        Returns:
            Numpy array with size of (self.batch_size, self.target_size,
            self.target_size, self.CH)
        """
        output = self._queue.get()
        return output


    def cropPatch(self, image, topleft_loc, crop_size):
        """Crop an image to a patch.

        Args:
            image: Numpy array to crop.
            topleft_loc: (topleft.y, topleft.x)
            crop_size: (height, width)

        Returns:
            Numpy array with size of (self.target_size, self.target_size, self.CH)
        """
        return image[topleft_loc[0]:topleft_loc[0]+crop_size[0],
                topleft_loc[1]:topleft_loc[1]+crop_size[1]]
        
    def applyRandomTransform(self, imageA, imageB=None, reshape=False):
        """Apply random transform for data augmentation.
        If imageB is provided, do the exactly same operation to make pair data.

        Args:
            imageA: 
            imageB:
            reshape: Option to provide for scipy.ndimage.interpolation.rotate.

        Returns:
            One or pair of numpy array. 
        """
        rand_degree = np.random.randint(0, 360)
        rand_flip = np.random.randint(0, 2)
        if rand_flip:
                imageA = np.flip(imageA, 1)
        imageA = scipy.ndimage.interpolation.rotate(imageA, rand_degree, order=0,
            cval=255.0, reshape=reshape)
        if imageB is None:
            return imageA
        else: # Apply same random transform
            if rand_flip:
                    imageB = np.flip(imageB, 1)
            imageB = scipy.ndimage.interpolation.rotate(imageB, rand_degree, order=0,
                cval=255.0, reshape=reshape)
            return imageA, imageB

    def randomTopleft(self, image, target_size):
        """Pick random but valid position of a topleft point for cropping.
        """
        H, W = image.shape[0], image.shape[1]
        Y_SZ, X_SZ = target_size
        if H > Y_SZ:
            crop_height = Y_SZ
        else:
            crop_height = H
        if W > X_SZ:
            crop_width = X_SZ
        else:
            crop_width = W
        topleft_x = np.random.randint(0, W - crop_width+1)
        topleft_y = np.random.randint(0, H - crop_height+1)
        return (topleft_y, topleft_x), (crop_height, crop_width)


if __name__ == '__main__':
        pass
