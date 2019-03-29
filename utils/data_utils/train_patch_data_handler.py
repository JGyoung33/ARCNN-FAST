import cv2
import os
import numpy as np
from utils.data_utils.train_data_handler import TrainDataHandler


class TrainPatchDataHandler(TrainDataHandler):
    """Data handler that manages patch image data.

    Args:
        data_dir: Dir for data
        batch_size: batch size
        target_size: image patch size
        is_color: decide the number of channel (1 or 3)
        random_resize: If True, it first resizes each image before cropping
                     for the augmentation
        num_patch: The number of patches to provide per each image.
        color_augment: If true, do color/noise augmentation
        num_threads: the number of thread to use for generating data
                   (please use < 5 threads)
    """

    def __init__(self, data_dir, batch_size, target_size, is_color=False,
                 random_resize=False, num_patch=1, color_augment=False,
                 num_threads=1, seed=0):
        super(TrainPatchDataHandler, self).__init__(batch_size, target_size,
                                                    is_color, num_threads)

        self.seed = seed
        self.data_dir = data_dir
        self.color_augment = color_augment
        self.image_paths = self.collectImagePaths(data_dir)
        self.num_patch = num_patch
        self.random_resize = random_resize
        if not num_threads < 5:
            raise AssertionError("trying too many threads (you should use thread less than 5")

        self.startThreads()

    def collectImagePaths(self, data_dir):
        """Collect image files from the data_dir and return the list.

        Returns:
            list of image paths in data_dir.
        """
        img_list = []  # Return value

        for (path, dir, files) in os.walk(data_dir):
            for filename in files:
                basename, ext = os.path.splitext(filename)
                if ext == '.png' or ext == '.jpg':
                    img_list.append(os.path.join(path, filename))

        return img_list

    def _enqueue_op(self, queue, msg_queue, random_seed):
        """Main function that generates a batch of data.
        Each thread runs this function to generate a batch and enqueue it
        to the global queue. The inheritted function "next" from
        train_data_handler will dequeue each data to feed the network.
        """
        np.random.seed(random_seed + self.seed)  # To make each thread get different data.

        while msg_queue.qsize() == 0:
            BS = self.batch_size
            Y_SZ = self.target_size
            X_SZ = self.target_size
            bs_patches = []

            for i in range(self.num_patch):
                bs_patches.append(np.ones([BS, Y_SZ, X_SZ, self.CH]))

            for i in range(self.num_patch):
                for j in range(BS):
                    index = np.random.randint(0, len(self.image_paths))
                    image = cv2.imread(self.image_paths[index], self.is_color)

                    if self.is_color:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.random_resize:  # Augmentation
                        random_ratio = np.random.randint(5, 15) / 10.0
                        random_scale = max(0.5, np.random.randint(1, 20) / 10.0)
                        random_fx = random_ratio * random_scale
                        random_fy = random_scale / random_ratio
                        fx = int(image.shape[1] * random_fx)
                        fy = int(image.shape[0] * random_fy)
                        image = cv2.resize(image, (fx, fy))

                    if not self.is_color:
                        image = np.reshape(image, [image.shape[0], image.shape[1], 1])

                    image = self.applyRandomTransform(image, reshape=False)

                    patch = self.getRandomPatch(image, [Y_SZ, X_SZ])

                    patch = patch / 255.0

                    if self.color_augment:
                        patch = setRandomWhiten(patch)
                        patch = setRandomBackground(patch)

                    bs_patches[i][j] = patch

            if self.num_patch == 1:  # Return single element(Not List)
                bs_patches = bs_patches[0]

            queue.put(bs_patches)

    def getRandomPatch(self, image, target_size):
        """Crop random patch from the given image with target_size.
        """
        Y_SZ, X_SZ = target_size
        topleft_loc, crop_size = self.randomTopleft(image, target_size)
        output = np.ones([Y_SZ, X_SZ, self.CH]) * 255
        output[0:crop_size[0], 0:crop_size[1], :] = \
            self.cropPatch(image, topleft_loc, crop_size)

        return output


if __name__ == '__main__':
    pass
