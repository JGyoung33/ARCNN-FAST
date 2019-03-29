import random
import sys
import numpy as np
import os
import cv2

class TestDataHandler(object):
    """Test data handler that provides one image per one next call.
    """
    def __init__(self, root_path, max_size=1024, use_fixed_size=False, is_color=False,
            binarize=False):

        self._image_paths = []
        self._index = 0
        self.max_size = max_size
        self.is_color = is_color
        self.use_fixed_size = use_fixed_size
        self.binarize = binarize
        if is_color:
            self.CH = 3
        else:
            self.CH = 1

        self._get_image_paths(root_path)
        self._total_num = len(self._image_paths)

    def _get_image_paths(self, root_path):
        for (path, dir, files) in os.walk(root_path):
            for filename in files:
                basename, ext = os.path.splitext(filename)
                if ext == '.jpg' or ext == '.png' or ext == '.jpeg':
                    self._image_paths.append(os.path.join(path, filename))

    def get_total_num(self):
        return self._total_num

    def next(self):
        """Load and return the next image. If the image is over the max_size,
        resize first. If there is no more image, return None

        Returns:
            output: Numpy array with size of (1, resized_height, resized_width, CH)
            original_size: shape of the original image.
            image_path: Refer for generate output filename.

        """
        if self._total_num <= self._index:
            return None

        image = cv2.imread(self._image_paths[self._index], self.is_color).astype(np.float32)
        if self.is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_size = image.shape
        bigger_size = max(original_size[0], original_size[1])

        if bigger_size > self.max_size:
            scale = self.max_size / float(bigger_size)
            image = cv2.resize(image, (int(original_size[1]*scale), int(original_size[0]*scale)))

        rows, cols = image.shape[0], image.shape[1]
        if rows % 32 != 0:
            rows += 32 - (rows%32)
        if cols % 32 != 0:
            cols += 32 - (cols%32)

        image = cv2.resize(image, (int(cols), int(rows)))
        image =  image / 255.0
        if self.is_color :
            CH = 3
        else:
            CH = 1

        if self.binarize:
            image[image > 0.9] = 1.0
            image[image <= 0.9] = 0.0

        if self.use_fixed_size:
            output = np.ones([1, self.max_size, self.max_size, CH])
            output[:, 0:rows, 0:cols, :] = image
        else:
            output = np.reshape(image, [1, rows, cols, CH])

        self._index += 1

        return output, original_size, self._image_paths[self._index-1]


    def rand_pick(self,seed =None):
        if seed != None : seed = time.time()
        else : seed = seed

        rand_impath = random.choice(self._image_paths)
        image = cv2.imread(rand_impath, self.is_color).astype(np.float32)
        if self.is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_size = image.shape
        bigger_size = max(original_size[0], original_size[1])

        if bigger_size > self.max_size:
            scale = self.max_size / float(bigger_size)
            image = cv2.resize(image, (int(original_size[1] * scale), int(original_size[0] * scale)))

        rows, cols = image.shape[0], image.shape[1]
        if rows % 32 != 0:
            rows += 32 - (rows % 32)
        if cols % 32 != 0:
            cols += 32 - (cols % 32)

        image = cv2.resize(image, (int(cols), int(rows)))
        image = image / 255.0
        if self.is_color:
            CH = 3
        else:
            CH = 1

        if self.binarize:
            image[image > 0.9] = 1.0
            image[image <= 0.9] = 0.0

        if self.use_fixed_size:
            output = np.ones([1, self.max_size, self.max_size, CH])
            output[:, 0:rows, 0:cols, :] = image
        else:
            output = np.reshape(image, [1, rows, cols, CH])

        return output, original_size, rand_impath


"""=====================================================================================================================
                                            module test 
====================================================================================================================="""
if __name__ == "__main__":
    root_path = '/datasets/sketchDB/safebooru/safebooru_lines'
    DH = TestDataHandler(root_path)
    img,_,_= DH.rand_pick()
    print(img.shape)
    img = img.reshape(img.shape[1],img.shape[2])
    plt.axis("off")
    plt.imshow(img)
    plt.show()
