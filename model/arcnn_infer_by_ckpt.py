import tensorflow as tf
import numpy as np
from pprint import pprint
from glob import glob
import matplotlib.pyplot as plt
from scipy import misc
import time
import cv2
import os

class ARCNN_InferbyCKPT(object):
    def __init__(self, sess, model_path):
        self.sess = sess
        self.model_path = model_path
        self.initialize()

    def initialize(self):
        tf.train.import_meta_graph(self.model_path+".meta")
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="arcnn")
        new_saver = tf.train.Saver(var_list=vars, max_to_keep=0)
        new_saver.restore(self.sess,self.model_path)
        g=tf.get_default_graph()


        list = g.get_operations()
        for l in list :
            if l.name.__contains__("handler"):
                print(l)
        pprint(list)


        self.image_test_A = g.get_tensor_by_name("input_Test_A:0")
        self.image_test_B = g.get_tensor_by_name("input_Test_B:0")
        self.handler_tensor_output_recon = g.get_tensor_by_name("arcnn/output_recon:0")


    def inference_prepare(self,input_img):
        print("image shape : " , input_img.shape)

        return img_resized

    def inference_recon(self,input_img):
        img_resized = self.inference_prepare(input_img)
        result_img = self.sess.run(self.handler_tensor_output_recon, feed_dict={self.image_test_A: img_resized})
        return result_img




if __name__ == "__main__":
    with tf.Session() as sess:
        model_path = os.path.normcase("../asset/checkpoint/infer/ARCNN_default_04_01_17_35_08/model.ckpt-0")
        model = ARCNN_InferbyCKPT(sess=sess,model_path=model_path)

        data_list = sorted(glob(os.path.join(os.path.normcase('../dataset/test/Set5'),"*.*")))
        print("num of data :", len(data_list))

        for d in data_list:
            #read
            img = misc.imread(d)

            #preprocess
            h, w, c = img.shape
            h_ = h - h % 2
            w_ = w - w % 2
            img_resized = cv2.resize(img, (int(w_), int(h_))).reshape([1, int(h_), int(w_), c])
            if (np.max(img_resized) > 1): img_resized = (img_resized / 255).astype(np.float32)


            #inference
            start = time.time()
            infered_img = model.inference_recon(img_resized)
            print("shape:",img_resized.shape,"elapse:", time.time() - start)

            #visualize
            plt.axis("off")
            plt.imshow(np.concatenate([img_resized[0], infered_img[0]], axis=1))
            plt.show()


