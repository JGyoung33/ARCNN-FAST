import datetime
import sys
sys.path.append('../')
import argparse
import os
import pprint
import tensorflow as tf
from model.arcnn_build_model_for_train import build_model
from utils.data_utils.train_patch_data_handler import TrainPatchDataHandler
from utils.data_utils.test_data_handler import TestDataHandler
from utils.utils import *
from glob import glob

"""=====================================================================================================================
                                            Trainer
====================================================================================================================="""
def train(args, sess):
    dataTrain_handler = TrainPatchDataHandler(
        os.path.normcase('../dataset/train/BSD400'),
        args.batch_size, args.target_size, is_color=True,seed = 0)

    test_img_paths = sorted(glob(os.path.join(os.path.normcase('../dataset/test/Set5'),"*.*")))
    test_imgs = []
    print(test_img_paths)
    for img_path in test_img_paths:
        img = cv2.imread(img_path).astype(np.float32)[...,::-1]
        img /= 255.0
        h, w, c = img.shape
        h_ = h - h% 2
        w_ = w - w% 2
        img = cv2.resize(img, (int(w_), int(h_))).reshape([1, int(h_), int(w_), c])
        test_imgs.append(img)




    BS = args.batch_size
    SZ = args.target_size
    CH = 3 if args.is_color else 1

    # =============================== training  =======================================
    try:
        """ set placehodlers """
        input_A = tf.placeholder(tf.float32, shape=[BS, SZ, SZ, CH], name='input_A')
        input_B = tf.placeholder(tf.float32, shape=[BS, SZ, SZ, CH], name='input_B')

        input_TEST_A = tf.placeholder(tf.float32, shape=[1, None, None, CH], name='input_Test_A')
        input_TEST_B = tf.placeholder(tf.float32, shape=[1, None, None, CH], name='input_Test_B')



        global_step = tf.Variable(0, trainable=False)
        incre_global_step = tf.assign(global_step, global_step + 1)

        start_learning_rate = args.learning_rate
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 2000, 0.1, staircase=True)


        """ build model """
        print("Build model graph...")
        build_model_template = tf.make_template('scale_by_y', build_model, learning_rate = learning_rate, args=args)
        [train_op, scalars, images] = build_model_template(input_A, input_B, learning_rate = learning_rate, args=args,)
        [_, scalars_test, images_test] = build_model_template(input_TEST_A, input_TEST_B, learning_rate = learning_rate, args=args,)

        checkpoint_dir_train = os.path.join(args.checkpoint_dir, "train")
        checkpoint_dir_test = os.path.join(args.checkpoint_dir, "test")
        summary_writer = tf.summary.FileWriter(checkpoint_dir_train, graph=sess.graph)
        summary_op = select_summary(learning_rate, images, scalars)
        summary_writer_test = tf.summary.FileWriter(checkpoint_dir_test, graph=sess.graph)
        summary_op_test = select_summary(learning_rate, images_test, scalars_test)




        """ init model """
        model_saver = tf.train.Saver(max_to_keep=100)
        tf.global_variables_initializer().run()
        step = restore_model(args, sess)
        tf.assign(global_step,step)


        """ prepare fetch_dict """
        fetch_dict = {
            "G_op": train_op["G_op"],
            "loss": scalars["loss"],
            "psnr": scalars["psnr"],
            "gstep": incre_global_step,
        }

        fetch_dict_test = {
            "loss": scalars_test["loss"],
            "psnr": scalars_test["psnr"],
        }

        while True: # We manually shutdown

            """ run """
            dataTrain = dataTrain_handler.next()
            dataA,bytes_list = cvt_jpeg(dataTrain)
            dataB = dataTrain
            result = sess.run(fetch_dict, feed_dict={input_A: dataA, input_B: dataB})


            """ post_processing """
            # Print log
            if global_step.eval() % 10 == 0:
                print("Iteration %d : loss %f psrn %f" % (global_step.eval(), result["loss"], result["psnr"]))

            if global_step.eval() % 10 == 0:
                psnr = []
                elapse = []
                for test_img in test_imgs:

                    start = time.time()
                    dataTestA,bytes_list = cvt_jpeg(test_img)
                    dataTestB = test_img

                    result_test = sess.run(fetch_dict_test, feed_dict={input_TEST_A: dataTestA, input_TEST_B: dataTestB})
                    psnr.append(result_test["psnr"])
                    elapse.append(time.time() - start)
                print("Iteration %d : loss %f psnr_test %f elapse %f" % (global_step.eval(), result_test["loss"], np.mean(psnr), np.mean(elapse)))


            # write summary
            if global_step.eval() % 50 == 0:
                fetch_dict_summary = {"summary": summary_op}
                result_summary = sess.run(fetch_dict_summary, feed_dict={input_A: dataA, input_B: dataB})
                summary_writer.add_summary(result_summary["summary"], global_step.eval())
                summary_writer.flush()


                dataTest = test_imgs[-1]
                dataTestA, bytes_list = cvt_jpeg(dataTest)
                dataTestB = dataTest

                fetch_dict_summary_test = {"summary": summary_op_test}
                result_summary_test = sess.run(fetch_dict_summary_test, feed_dict={input_TEST_A: dataTestA, input_TEST_B: dataTestB})
                summary_writer_test.add_summary(result_summary_test["summary"], global_step.eval())
                summary_writer_test.flush()


            # save model
            if global_step.eval() % 5 == 0:
                save_path = model_saver.save(sess, os.path.join(args.checkpoint_dir, "train", "model.ckpt"), global_step=global_step.eval())



    finally:
        dataTrain_handler.kill()



"""=====================================================================================================================
                                            module test 
====================================================================================================================="""
if __name__ == '__main__':
# =======================================================
# [global variables]
# =======================================================
    pp = pprint.PrettyPrinter()
    args = None
    DATA_PATH = "./train/"
    TEST_DATA_PATH = "./data/cookbook/"

# =======================================================
# [add parser]
# =======================================================
    parser = argparse.ArgumentParser()
    #===================== common configuration ============================================
    parser.add_argument("--exp_tag", type=str, default="ARCNN tensorflow. Implemented by Dohyun Kim")
    parser.add_argument("--model_name", type=str, default="ARCNN")
    parser.add_argument("--model_tag", type=str, default="default")
    parser.add_argument("--gpu", type=int, default=0)  # -1 for CPU
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_size", type=int, default=64)
    parser.add_argument("--is_color", type=bool, default=True)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--jpgqfactor", type= int, default =60)
    parser.add_argument("--train_subdir", default="BSD400")
    parser.add_argument("--test_subdir", default="Set5")
    parser.add_argument("--g_type", type=int, default=3)  # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--lr_decay_rate", type=float, default=1e-1)
    parser.add_argument("--lr_step_size", type=int, default=20)  # 9999 for no decay
    parser.add_argument("--checkpoint_dir", default="../asset/checkpoint")



    print("=====================================================================")
    args = parser.parse_args()
    print("Eaxperiment tag : " + args.exp_tag)
    pp.pprint(args)

    time_now = datetime.datetime.now()
    name = "%s_%s_%02d_%02d_%02d_%02d_%02d" % (args.model_name, args.model_tag,
                                               time_now.month, time_now.day, time_now.hour, time_now.minute,
                                               time_now.second)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, name)
    check_folder(os.path.join(args.checkpoint_dir, "train"))
    check_folder(os.path.join(args.checkpoint_dir, "test"))
    print("=====================================================================")



    tf.set_random_seed(123)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_frac
    # config.allow_soft_placement = True


    """do task """
    with tf.Session(config=config) as sess:
        train(args, sess)






















'''
if __name__ == '__main__':
# =======================================================
# [global variables]
# =======================================================
    pp = pprint.PrettyPrinter()
    args = None
    DATA_PATH = "./train/"
    TEST_DATA_PATH = "./data/cookbook/"

# =======================================================
# [add parser]
# =======================================================
    parser = argparse.ArgumentParser()
    #===================== common configuration ============================================
    parser.add_argument("--exp_tag", type=str, default="ARCNN tensorflow. Implemented by Dohyun Kim")
    parser.add_argument("--gpu", type=int, default=0)  # -1 for CPU

    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_size", type=int, default=55)
    parser.add_argument("--is_color", type=bool, default=False)

    parser.add_argument("--stride_size", type=int, default=20)
    parser.add_argument("--deconv_stride", type = int, default = 2)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--jpgqfactor", type= int, default =60)

    parser.add_argument("--train_subdir", default="BSD400")
    parser.add_argument("--test_subdir", default="Set5")
    parser.add_argument("--infer_imgpath", default="monarch.bmp")  # monarch.bmp
    parser.add_argument("--type", default="YCbCr", choices=["RGB","Gray","YCbCr"])#YCbCr type uses images preprocessesd by matlab
    parser.add_argument("--c_dim", type=int, default=3) # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)
    parser.add_argument("--mode", default="train", choices=["train", "cookbook", "inference", "test_plot"])

    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_decay_rate", type=float, default=1e-1)
    parser.add_argument("--lr_step_size", type=int, default=20)  # 9999 for no decay
    parser.add_argument("--checkpoint_dir", default="checkpoint")
    parser.add_argument("--cpkt_itr", default=0)  # -1 for latest, set 0 for training from scratch
    parser.add_argument("--save_period", type=int, default=1)

    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--save_extension", default=".jpg", choices=["jpg", "png"])

    print("=====================================================================")
    args = parser.parse_args()
    if args.type == "YCbCr":
        args.c_dim = 1; #args.train_subdir += "_M"; args.test_subdir += "_M"
    elif args.type == "RGB":
        args.c_dim = 3;
    elif args.type == "Gray":
        args.c_dim = 1
    print("Eaxperiment tag : " + args.exp_tag)
    pp.pprint(args)
    print("=====================================================================")

# =======================================================
# [make directory]
# =======================================================
    if not os.path.exists(os.path.join(os.getcwd(), args.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), args.checkpoint_dir))
    if not os.path.exists(os.path.join(os.getcwd(), args.result_dir)):
        os.makedirs(os.path.join(os.getcwd(), args.result_dir))

# =======================================================
# [Main]
# =======================================================
    # -----------------------------------
    # system configuration
    # -----------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    if args.gpu == -1: config.device_count = {'GPU': 0}
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.operation_timeout_in_ms=10000


    # -----------------------------------
    # build model
    # -----------------------------------
    with tf.Session(config = config) as sess:
        arcnn = ARCNN_FAST(sess = sess, args = args)

        # -----------------------------------
        # train, cookbook, inferecnce
        # -----------------------------------
        if args.mode == "train":
            arcnn.train()


        elif args.mode == "cookbook":
            arcnn.test()


        elif args.mode == "inference":
            pass

        elif args.mode == "test_plot":
            pass

'''

