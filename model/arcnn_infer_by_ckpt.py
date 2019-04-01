import tensorflow as tf
import numpy as np
from pprint import pprint
from glob import glob
import matplotlib.pyplot as plt
from scipy import misc
import time
import cv2

class SketchSimplifier_InferbyCKPT(object):
    def __init__(self, sess, model_path):
        self.sess = sess
        self.model_path = model_path
        self.initialize()

    def initialize(self):
        tf.train.import_meta_graph(self.model_path+".meta")
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="sketch_simplifier")  + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="L2_line_normalizer")
        new_saver = tf.train.Saver(var_list=vars, max_to_keep=0)
        new_saver.restore(self.sess,self.model_path)
        g=tf.get_default_graph()

        '''
        list = g.get_operations()
        for l in list :
            if l.name.__contains__("handler"):
                print(l)
        #pprint(list)
        '''

        self.image_test = g.get_tensor_by_name("input_A:0")
        self.image_test2 = g.get_tensor_by_name("input_D:0")

        self.handler_tensor_line_normalizer = g.get_tensor_by_name("sketch_simplifier/handler_tensor_line_normalizer:0")
        self.handler_tensor_binarizer = g.get_tensor_by_name("sketch_simplifier/handler_tensor_binarizer:0")
        self.handler_tensor_major_simplifier = g.get_tensor_by_name("sketch_simplifier/handler_tensor_major_simplifier:0")
        self.handler_tensor_minor_simplifier = g.get_tensor_by_name("sketch_simplifier/handler_tensor_minor_simplifier:0")
        self.handler_tensor_L2_line_normalizer = g.get_tensor_by_name("L2_line_normalizer/handler_tensor_L2_line_normalizer:0")



    def inference_prepare(self,input_img):
        print("image shape : " , input_img.shape)

        s = 1.0
        h, w = input_img.shape
        h_ = h * s - h * s % 8
        w_ = w * s - w * s % 8
        img_resized = cv2.resize(input_img.reshape([h, w]), (int(w_), int(h_))).reshape([1, int(h_), int(w_), 1])
        return img_resized

    def inference_line_normalize(self,input_img):
        img_resized = self.inference_prepare(input_img)
        result_img = self.sess.run(self.handler_tensor_line_normalizer, feed_dict={self.image_test: img_resized})

        result_img = result_img[0]
        return result_img

    def inference_binarize(self,input_img):
        img_resized = self.inference_prepare(input_img)
        result_img = self.sess.run(self.handler_tensor_binarizer, feed_dict={self.image_test: img_resized})

        result_img = result_img[0,:,:,0]
        return result_img

    def inference_major_simplification(self,input_img):
        img_resized = self.inference_prepare(input_img)
        result_img = self.sess.run(self.handler_tensor_major_simplifier, feed_dict={self.image_test: img_resized})

        result_img = result_img[0,:,:,0]
        return result_img

    def inference_minor_simplification(self,input_img):
        img_resized = self.inference_prepare(input_img)
        result_img = self.sess.run(self.handler_tensor_minor_simplifier, feed_dict={self.image_test: img_resized})

        result_img = result_img[0,:,:,0]
        return result_img

    def inference_L2_line_normalizer(self,input_img):
        img_resized = self.inference_prepare(input_img)
        result_img = self.sess.run(self.handler_tensor_L2_line_normalizer, feed_dict={self.image_test2: img_resized})

        result_img = result_img[0,:,:,0]
        #if result_img.shape[-1] <3 : result_img = np.concatenate([result_img,result_img,result_img],axis=-1) #force to color
        return result_img






if __name__ == "__main__":
    with tf.Session() as sess:
        model_path = "/checkpoints/sketch_simplifier/model_fordemo.ckpt-0"
        model = SketchSimplifier_InferbyCKPT(sess=sess,model_path=model_path)
        num_latent = 8

        plt.axis("off")
        data_list = glob("/datasets/sketchDB/safebooru/safebooru_lines_test/*.*")
        print("num of data :", len(data_list))

        cv2.namedWindow('img_veiwer', 0)
        cv2.resizeWindow('img_veiwer', 500*2, 500)

        img = misc.imread(data_list[0], mode='L')
        if (np.max(img) > 1): img = (img / 255).astype(np.float32)
        infered_img = img

        while True :
            h, w = infered_img.shape
            img_resized = cv2.resize(img,(int(w), int(h)))
            cv2.imshow('img_veiwer', np.concatenate([img_resized,infered_img],axis=-1))

            if cv2.waitKey(10) & 0xFF == ord('w'):
                break

            print("[b : binarization, m : major simplification, n : minor simplification, 1 : L1 line normlization, 2 : L2 line normlization, r : reset")
            key = input()
            if key == '1':
                infered_img = model.inference_line_normalize(infered_img)

            elif key == '2':
                infered_img = model.inference_L2_line_normalizer(infered_img)

            elif key == 'b':
                infered_img = model.inference_binarize(infered_img)

            elif key == 'm':
                infered_img = model.inference_major_simplification(infered_img)

            elif key == 'n':
                infered_img = model.inference_minor_simplification(infered_img)

            elif key == 'r':
                infered_img = img

            else:
                print("invaild command")
                continue


