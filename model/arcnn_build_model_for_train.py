import argparse
import sys
sys.path.append('../')
import os
import time
import tensorflow as tf
from utils.imresize import *
from utils.metrics import *
import numpy as np
from tqdm import tqdm
from utils.cookbook.ops import *
from utils.cookbook.utils import *
from functools import partial


""""================================================================
* modules
================================================================="""
def inner_model(x, scope_name, reuse, is_color=False, is_training=True, output_activation=tf.nn.sigmoid, norm_type=["instance_norm"], verbose = False):
    if not is_color:        image_channel = 1
    else:                   image_channel = 3

    with tf.variable_scope(scope_name, reuse=reuse) as vscope:
        input = x
        with tf.variable_scope("feature_extraction", reuse=reuse) as scope:
            x = conv(x, 64, kernel= 9, stride= 1, scope = "conv_0")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("feature_enhancement", reuse=reuse) as scope:
            x = conv(x, 32, kernel= 1, stride= 2, scope = "conv_0")
            x = lrelu(x, alpha=0.1)

            x = conv(x, 32, kernel= 7, stride= 1, scope = "conv_1")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("mapping", reuse=reuse) as scope:
            x = conv(x, 64, kernel= 1, stride= 1, scope = "conv_0")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("reconstruction", reuse=reuse) as scope:
            x = deconv(x, image_channel, kernel=9, stride=2, scope="deconv_0")
            if verbose: print(x)

        output = x + input
        if verbose: print(output)

    return output




def inner_model2(x, scope_name, reuse, is_color=False, is_training=True, output_activation=tf.nn.sigmoid, norm_type=["instance_norm"], verbose = False):
    if not is_color :  image_channel = 1
    else            :  image_channel = 3

    with tf.variable_scope(scope_name, reuse=reuse) as vscope:
        input = x
        with tf.variable_scope("feature_extraction", reuse=reuse) as scope:
            for i in range(4):
                x = conv(x, 64, kernel= 3, stride= 1, scope = "conv_{}".format(i))
                x = lrelu(x, alpha=0.1)

        with tf.variable_scope("reconstruction", reuse=reuse) as scope:
            x = conv(x, image_channel, kernel= 3, stride= 1, scope = "conv_last")
            output = x + input
    return output


def inner_model3(x, scope_name, reuse, is_color=False, is_training=False, output_activation=tf.nn.sigmoid, norm_type=["instance_norm"], verbose = False):
    if not is_color:        image_channel = 1
    else:                   image_channel = 3

    with tf.variable_scope(scope_name, reuse=reuse) as vscope:
        input = x
        with tf.variable_scope("feature_extraction", reuse=reuse) as scope:
            x = conv(x, 64, kernel= 9, stride= 1, scope = "conv_0")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("feature_enhancement", reuse=reuse) as scope:
            x = conv(x, 32, kernel= 1, stride= 2, scope = "conv_0")
            x = lrelu(x, alpha=0.1)

            x = conv(x, 32, kernel= 7, stride= 1, scope = "conv_1")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("mapping", reuse=reuse) as scope:
            x = conv(x, 64, kernel= 1, stride= 1, scope = "conv_0")
            x = lrelu(x, alpha=0.1)

        with tf.variable_scope("reconstruction", reuse=reuse) as scope:
            x = conv(x, 4, kernel= 1, stride= 1, scope = "conv_0")
            x = tf.depth_to_space(x, block_size=2)
            if verbose: print(x)

        output = x + input
        if verbose: print(output)
    return output



def inner_model4(x, scope_name, reuse, is_color=False, is_training=False, output_activation=tf.nn.sigmoid, norm_type=["instance_norm"], verbose = False):
    if not is_color:        image_channel = 1
    else:                   image_channel = 3

    with tf.variable_scope(scope_name, reuse=reuse) as vscope:
        input = x
        with tf.variable_scope("feature_extraction", reuse=reuse) as scope:
            x = conv(x, 64, kernel= 9, stride= 1, scope = "conv_0")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("feature_enhancement", reuse=reuse) as scope:
            x = conv(x, 32, kernel= 1, stride= 2, scope = "conv_0")
            x = lrelu(x, alpha=0.1)

            x = conv(x, 32, kernel= 7, stride= 1, scope = "conv_1")
            x = lrelu(x, alpha=0.1)
            if verbose :print(x)

        with tf.variable_scope("mapping", reuse=reuse) as scope:
            x = conv(x, 64, kernel= 1, stride= 1, scope = "conv_0")
            x = lrelu(x, alpha=0.1)

        with tf.variable_scope("reconstruction", reuse=reuse) as scope:
            x = resize(x)
            x = conv(x, 64, kernel= 1, stride= 1, scope = "conv_0")

        output = x + input
        if verbose: print(output)
    return output


""""================================================================
* Build model 
================================================================="""
def build_model(input_A,input_B, learning_rate, args=None):
    if args.g_type == 1:
        p_arcnn = partial(inner_model, is_color=False, is_training=True)
    elif args.g_type == 2:
        p_arcnn = partial(inner_model2, is_color=False, is_training=True)
    elif args.g_type == 3:
        p_arcnn = partial(inner_model3, is_color=False, is_training=True)
    elif args.g_type == 4:
        p_arcnn = partial(inner_model4, is_color=False, is_training=True)

    """ for return """
    images = None
    train_op = None
    scalars = None

    with tf.variable_scope("arcnn") as scope:
        #=============================== modules =======================================
        input_A_yuv = tf.image.rgb_to_yuv(input_A)
        input_A_y, input_A_uv = tf.split(input_A_yuv, [1,2], -1)
        input_B_yuv = tf.image.rgb_to_yuv(input_B)
        input_B_y, input_B_uv = tf.split(input_B_yuv, [1,2], -1)
        input_A_y_rec = p_arcnn(input_A_y, scope_name = "generator", reuse=False)


        # =============================== losses =======================================
        """ loss - supervised loss """
        L1_loss = tf.reduce_mean(tf.abs(input_A_y_rec - input_B_y))  # L1 is betther than L2

        """ merge losses """
        loss = L1_loss


        # ============================= optimizeres =====================================
        """ mark trainable varables and set train_op"""
        t_vars = tf.trainable_variables()
        print("========= Trainable variables ============")
        for v in t_vars: print(v)
        print("==========================================")

        #seperated for handling adversarial loss. but it is not necsessary in this project
        G_vars = [var for var in t_vars if "generator" in var.name]
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            G_op = optimizer.minimize(loss, var_list=G_vars)


        # =============================== return dicts  =======================================
        input_A_rec = tf.clip_by_value(tf.image.yuv_to_rgb(tf.concat([input_A_y_rec,input_A_uv],axis=-1)), clip_value_max=1, clip_value_min=0)

        images = {
            "result": tf.concat([input_A, input_A_rec, input_B],axis=1),
        }

        train_op = {
            "G_op": G_op,
        }

        scalars = {
            "loss": loss,
            "psnr_y": tf.reduce_mean(tf.image.psnr(input_A_y, input_B_y, max_val=1.0)),
            "psnr": tf.reduce_mean(tf.image.psnr(input_A_rec, input_B, max_val=1.0))
        }

    return train_op, scalars, images



""""================================================================
* Module test 
================================================================="""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #===================== common configuration ============================================
    parser.add_argument("--exp_tag", type=str, default="ARCNN tensorflow. Implemented by Dohyun Kim")
    parser.add_argument("--gpu", type=int, default=0)  # -1 for CPU

    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_size", type=int, default=64)
    parser.add_argument("--is_color", type=bool, default=True)

    parser.add_argument("--stride_size", type=int, default=20)
    parser.add_argument("--deconv_stride", type = int, default = 2)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--jpgqfactor", type= int, default =60)

    parser.add_argument("--train_subdir", default="BSD400")
    parser.add_argument("--test_subdir", default="Set5")
    parser.add_argument("--infer_imgpath", default="monarch.bmp")  # monarch.bmp
    parser.add_argument("--type", default="YCbCr", choices=["RGB","Gray","YCbCr"])#YCbCr type uses images preprocessesd by matlab
    parser.add_argument("--c_dim", type=int, default=3) # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)
    parser.add_argument("--g_type", type=int, default=3)  # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)

    parser.add_argument("--mode", default="train", choices=["train", "cookbook", "inference", "test_plot"])

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--lr_decay_rate", type=float, default=1e-1)
    parser.add_argument("--lr_step_size", type=int, default=20)  # 9999 for no decay
    parser.add_argument("--checkpoint_dir", default="../asset/checkpoint")
    parser.add_argument("--cpkt_itr", default=0)  # -1 for latest, set 0 for training from scratch
    parser.add_argument("--save_period", type=int, default=1)

    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--save_extension", default=".jpg", choices=["jpg", "png"])



    print("=====================================================================")
    args = parser.parse_args()

    BS,SZ,SZ,CH = (4,512,512,3)
    input_A = tf.placeholder(tf.float32, shape=[BS, SZ, SZ, CH], name='input_A')
    input_B = tf.placeholder(tf.float32, shape=[BS, SZ, SZ, CH], name='input_B')
    build_model(input_A,input_B, learning_rate=0.1, args = args)





"""

class ARCNN_FAST(object):
# ==========================================================
# class initializer
# ==========================================================
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.preprocess()
        self.model()
        self.other_tensors()
        self.init_model()


# ==========================================================
# preprocessing
# ==========================================================
    def preprocess(self):
        self.train_label = []
        self.train_input = []
        self.test_label = []
        self.test_input = []
        if self.args.type == "YCbCr" : input_setup = input_setup_demo
        elif self.args.type == "RGB" : input_setup = input_setup_demo
        else : input_setup = input_setup_demo

        # scale augmentation
        scale_temp = self.args.scale
        for s in [1, 0.9, 0.8, 0.7, 0.6]:
            self.args.scale = s
            train_input_, train_label_ = input_setup(self.args, mode="train")
            self.train_label.extend(train_label_)
            self.train_input.extend(train_input_)
        self.args.scale = scale_temp

        # augmentation (rotation, miror flip)
        self.train_label = augumentation(self.train_label)
        self.train_input = augumentation(self.train_input)

        # setup cookbook data
        self.test_input, self.test_label = input_setup(self.args, mode="cookbook")
        pass



# ==========================================================
# build model
# ==========================================================
    def model(self):
        with tf.variable_scope("ARCNN_FAST") as scope:
            shared_inner_model_template = tf.make_template('shared_model', self.inner_model)
            #self.images = tf.placeholder(tf.float32, [None, self.args.patch_size, self.args.patch_size, self.args.c_dim],  name='images')

            self.images = tf.placeholder(tf.float32, [None, self.args.patch_size, self.args.patch_size, self.args.c_dim],  name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.args.patch_size, self.args.patch_size, self.args.c_dim],  name='labels')
            self.pred = shared_inner_model_template(self.images)

            #self.image_test = tf.placeholder(tf.float32, [1, None, None, self.args.c_dim], name='images_test')
            self.image_test = tf.placeholder(tf.float32, [1, None, None, self.args.c_dim], name='image_test')
            self.label_test = tf.placeholder(tf.float32, [1, None, None, self.args.c_dim], name='labels_test')
            self.pred_test = shared_inner_model_template(self.image_test)





# ===========================================================
# inner model
# ===========================================================

    # ----------------------------------------------------------------------------------------





# ============================================================
# other tensors related with training
# ============================================================
    def other_tensors(self):
        with tf.variable_scope("trainer") as scope:
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.loss = tf.reduce_mean(tf.square(self.pred - self.labels))  # L1 is betther than L2
            self.learning_rate = tf.maximum(tf.train.exponential_decay(self.args.base_lr, self.global_step,
                                                                           len(self.train_label) // self.args.batch_size * self.args.lr_step_size,
                                                                           self.args.lr_decay_rate,
                                                                           staircase=True),
                                                self.args.min_lr)  # stair case showed better result

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            # tensor board
            self.summary_writer = tf.summary.FileWriter("./board", self.sess.graph)
            self.loss_history = tf.summary.scalar("loss", self.loss)
            self.summary = tf.summary.merge_all()
            self.psnr_history = []
            self.ssim_history = []



# ============================================================
# init tensors
# ============================================================
    def init_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)
        if self.cpkt_load(self.args.checkpoint_dir, self.args.cpkt_itr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def cpkt_save(self, checkpoint_dir, step):
        model_name = "checks.model"
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def cpkt_load(self, checkpoint_dir, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if checkpoint_itr == 0:
            print("train from scratch")
            return True

        elif checkpoint_dir == -1:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)

        else:
            ckpt = os.path.join(checkpoint_dir, "checks.model-" + str(checkpoint_itr))

        print(ckpt)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            return False





# ==========================================================
# functions
# ==========================================================
    def inference(self, input_img):
        if (np.max(input_img) > 1): input_img = (input_img / 255).astype(np.float32)

        size = input_img.shape
        if (len(input_img.shape) == 3):
            infer_image_input = input_img[:, :, 0].reshape(1, size[0], size[1], 1)
        else:
            infer_image_input = input_img.reshape(1, size[0], size[1], 1)

        sr_img = self.sess.run(self.pred_test, feed_dict={self.image_test: infer_image_input})
        # sr_img = np.expand_dims(sr_img,axis=-1)


        input_img = imresize(input_img,self.args.scale)
        if (len(input_img.shape) == 3):
            input_img[:, :, 0] = sr_img[0, :, :, 0]
        else:
            input_img = sr_img[0]

        return input_img #return as ycbcr




# ==========================================================
# train
# ==========================================================
    def train(self):
        self.test()
        print("Training...")
        start_time = time.time()


        for ep in range(self.args.epoch):
            # =============== shuffle and prepare batch images ============================
            seed = int(time.time())
            np.random.seed(seed); np.random.shuffle(self.train_label)
            np.random.seed(seed); np.random.shuffle(self.train_input)

            #================ train rec ===================================================
            batch_idxs = len(self.train_label) // self.args.batch_size
            for idx in tqdm(range(0, batch_idxs)):
                batch_labels = np.expand_dims(np.array(self.train_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size])[:,:,:,0],-1)
                batch_inputs = np.expand_dims(np.array(self.train_input[idx * self.args.batch_size: (idx + 1) * self.args.batch_size])[:,:,:,0],-1)

                feed = {self.images: batch_inputs, self.labels:batch_labels}
                _, err, lr, summary = self.sess.run( [self.train_op, self.loss, self.learning_rate, self.summary], feed_dict=feed)
                self.summary_writer.add_summary(summary,self.global_step.eval())



            #=============== print log =====================================================
            if ep % 1 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_com: [%.8f], lr: [%.8f]" \
                      % ((ep + 1), self.global_step.eval(), time.time() - start_time, np.mean(err), lr))
                self.test()


            #================ save checkpoints ===============================================
            if ep % self.args.save_period == 0:
                self.cpkt_save(self.args.checkpoint_dir, ep + 1)


# ==========================================================
# cookbook
# ==========================================================
    def test(self):
        print("Testing...")
        psnrs_preds = []
        ssims_preds = []

        preds = []
        labels = []
        images = []

        for idx in range(0, len(self.test_label)):
            test_label = np.array(self.test_label[idx]) #none,none,3
            test_input = np.array(self.test_input[idx])

            # === original =====
            for f in [5, 10, 20, 40, 50, 60, 80, 100]:
                cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori" + str(f) + ".jpg"),
                            (ycbcr2rgb(test_label)*255)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), f])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori.PNG"), (ycbcr2rgb(test_label)*255)[..., ::-1])
            # ==================

            result = self.inference(test_input)
            cv2.imwrite(os.path.join(self.args.result_dir,str(idx)+ "rec"+str(self.args.jpgqfactor)+".bmp"), (ycbcr2rgb(result)*255)[...,::-1])

            preds.append(result)
            labels.append(test_label)


        # cal PSNRs for each images upscaled from different depths
        for i in range(len(self.test_label)):
            if len(np.array(labels[i]).shape)==3 : labels[i] = np.array(labels[i])[:,:,0]
            if len(np.array(preds[i]).shape)==3 : preds[i] = np.array(preds[i])[:,:,0]
            psnrs_preds.append(psnr(labels[i], preds[i], max=1.0, scale=self.args.scale))
            ssims_preds.append(ssim(labels[i], preds[i], max=1.0, scale=self.args.scale))

        # print evalutaion results
        print("===================================================================================")
        print("PSNR: " + str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)) + "dB")
        print("SSIM: " + str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print("===================================================================================")

        self.psnr_history.append(str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)))
        self.ssim_history.append(str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print()
        for h in self.psnr_history:
            print(h, ",", end="")
        print()
        print()
        for h in self.ssim_history:
            print(h, ",", end="")
        print()
"""