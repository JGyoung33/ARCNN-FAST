import cv2
import os
import numpy as np
from utils.data_utils.train_data_handler import TrainDataHandler

class TrainPairDataHandler(TrainDataHandler):
    """Data handler that manages pair data.

    Args:
        dataA_dir: Dir for dataA. Note that the first splited string by "_" of
                 a basename of each file becomes a key value to find the pair 
                 data from dataB.
                 Also, each file needs to have a unique key value. 
                 example) dataAdir/0030_color0_hello.jpg => 0030 will be a key
        dataB_dir: Dir for dataB. Similarily, the first splited string by "_"
                 of a basename of each file used as a key to map with pair data.
                 Unlike dataA, each file can have duplicated key value. 
                 example) dataBdir/0030_sketch1.jpg, dataBdir/0030_sketch2.jpg
        batch_size: batch size
        target_size: image patch size
        is_color: decide the number of channel (1 or 3)
        random_resize: If True, it first resizes each image before cropping 
                     for the augmentation
        num_threads: the number of thread to use for generating data 
                   (please use < 5 threads)

    """

    def __init__(self, dataA_dir, dataB_dir, batch_size, target_size, 
            is_color=False, random_resize=True, num_threads=1):
        super(TrainPairDataHandler, self).__init__(batch_size, target_size,
                is_color, num_threads)

        self.random_resize = random_resize
        self.dataA_dir = dataA_dir
        self.dataB_dir = dataB_dir

        self.dataA_list = []
        self.dataB_list = []

        self.image_pair = self.collectImagePaths()

        if len(self.image_pair) < batch_size:
            mult = batch_size / len(self.image_pair) + 1
            self.image_pair = self.image_pair * int(mult)

        self.startThreads()


    def collectImagePaths(self):
        """The first splited string by "_" of a basename of each file becomes a 
        key value. A key of each file in dataA must be unique, but a key of
        one in dataB does not need to be unique.
        example) dataAdir/0030_line.jpg, dataAdir/0031_line.jpg, ...
        example) dataBdir/0030_sketch1.jpg, dataBdir/0030_sketch2.jpg, ...
        In the above case, dataA/0030_line1.jpg can not exist.

        Returns:
            list of pair image paths
        """
        dict_basename = {} # {key: basekey, value: [ paths ]}
        pair_list = [] # Return value (dataA, dataB)

        # Collect Key from dataA
        for (path, dirs, files) in os.walk(self.dataA_dir):
            for filename in files:
                basename, ext = os.path.splitext(filename)
                if ext == '.png' or ext == '.jpg':
                    basekey = basename.split("_")[0]
                    if basekey in dict_basename:
                        dict_basename[basekey].append(os.path.join(path, filename))
                    else:
                        dict_basename[basekey] = [ os.path.join(path, filename) ]

        # Collect dataB and make pair
        for (path, dir, files) in os.walk(self.dataB_dir):
            for filename in files:
                basename, ext = os.path.splitext(filename)
                if ext == '.png' or ext == '.jpg':
                    basekey = basename.split("_")[0]
                    dataA_paths = dict_basename[basekey]
                    for path_a in dataA_paths:
                        pair_list.append((path_a, os.path.join(path, filename)))
        print("data pair length : ", len(pair_list))

        return pair_list

    def _enqueue_op(self, queue, msg_queue, random_seed):
        """Main function that generates a batch of data.
        Each thread runs this function to generate a batch and enqueue it
        to the global queue. The inheritted function "next" from 
        train_data_handler will dequeue each data to feed the network.
        """
        np.random.seed(random_seed) # To make each thread get different data.

        while msg_queue.qsize() == 0:
            BS = self.batch_size
            SZ = self.target_size
            patchesA = np.ones([BS, SZ, SZ, self.CH])
            patchesB = np.ones([BS, SZ, SZ, self.CH])

            for i in range(self.batch_size):
                random_index = np.random.randint(0, len(self.image_pair))
                pathA = self.image_pair[random_index][0]
                pathB = self.image_pair[random_index][1]

                imageA = cv2.imread(pathA, self.is_color)
                imageB = cv2.imread(pathB, self.is_color)
                if self.is_color:
                    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
                    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
                    
                if imageA.shape != imageB.shape:
                    imageA = cv2.resize(imageA, (imageB.shape[1], imageB.shape[0]))

                if self.random_resize: # Augmentation
                    random_ratio = np.random.randint(5, 15) / 10.0 
                    random_scale = max(0.5, np.random.randint(1, 20) / 10.0)
                    random_fx = random_ratio * random_scale
                    random_fy = random_scale / random_ratio
                    fx = int(imageB.shape[1]*random_fx)
                    fy = int(imageB.shape[0]*random_fy)

                    imageA = cv2.resize(imageA, (fx, fy))
                    imageB = cv2.resize(imageB, (fx, fy))

                if not self.is_color:
                    imageA = np.reshape(imageA, [imageA.shape[0], imageA.shape[1], 1])
                    imageB = np.reshape(imageB, [imageB.shape[0], imageB.shape[1], 1])

                # Crop Random Place
                patchA, patchB = self.cropRandomPair(imageA, imageB, [SZ, SZ])
                patchA, patchB = self.applyRandomTransform(patchA, patchB)

                patchA /= 255.0
                patchB /= 255.0

                patchesA[i] = patchA
                patchesB[i] = patchB

            queue.put((patchesA, patchesB))


    def cropRandomPair(self, imageA, imageB, target_size):
        """Crop same position with same size from the pair data

        """
        Y_SZ, X_SZ = target_size
        topleft_loc, crop_size = self.randomTopleft(imageA, target_size)
        outputA = np.ones([Y_SZ, X_SZ, self.CH])*255
        outputB = np.ones([Y_SZ, X_SZ, self.CH])*255
        outputA[0:crop_size[0], 0:crop_size[1], :] = \
                self.cropPatch(imageA, topleft_loc, crop_size)
        outputB[0:crop_size[0], 0:crop_size[1], :] = \
                self.cropPatch(imageB, topleft_loc, crop_size)

        return outputA, outputB


