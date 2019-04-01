import sys
sys.path.append('../')
import argparse
import pprint
from old.arcnn_build_model_for_train import build_model
from utils.data_utils.train_patch_data_handler import TrainPatchDataHandler
from utils.data_utils.test_data_handler import TestDataHandler
from utils.utils import *

"""=====================================================================================================================
                                            Trainer
====================================================================================================================="""
def train(args, sess):
    dataTrain_handler = TrainPatchDataHandler(
        os.path.normcase('../dataset/train/BSD400'),
        args.batch_size, args.target_size, is_color=True,seed = 0)


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
        learning_rate = tf.train.exponential_decay(start_learning_rate,
                                                   global_step, 100000, 0.8, staircase=True)


        """ build model """
        print("Build model graph...")
        build_model_template = tf.make_template('scale_by_y', build_model, learning_rate = learning_rate, args=args)
        [train_op, scalars, images] = build_model_template(input_A, input_B, learning_rate = learning_rate, args=args,)
        [_, scalars_test, images_test] = build_model_template(input_TEST_A, input_TEST_B, learning_rate = learning_rate, args=args,)
        summary_writer = tf.summary.FileWriter(args.checkpoint_dir, graph=sess.graph)
        summary_op = select_summary(learning_rate, images, scalars)
        summary_writer_test = tf.summary.FileWriter(args.checkpoint_dir+"_test", graph=sess.graph)
        summary_op_test = select_summary(learning_rate, images_test, scalars_test)




        """ init model """
        model_saver = tf.train.Saver(max_to_keep=100)
        tf.global_variables_initializer().run()
        #step = restore_model(args, sess)


        while True: # We manually shutdown
            dataTrain = dataTrain_handler.next()
            dataTrain_ycbcr = rgb2ycbcr_batch(dataTrain)
            dataB = dataTrain_ycbcr[:,:,:,0:1]

            dataTrain_jpeg = cvt_jpeg(dataTrain)
            dataTrain_jpeg_ycbcr = rgb2ycbcr_batch(dataTrain_jpeg)
            dataA = dataTrain_jpeg_ycbcr[:,:,:,0:1]


            #plt.imshow(np.concatenate([dataA,dataB],axis=2)[2,:,:,0],cmap='gray')
            #plt.show()
            #plt.imshow(np.concatenate([dataA,dataB],axis=2)[3,:,:,0],cmap='gray')
            #plt.show()

            #print(dataTrain.dtype,np.max(dataTrain))
            #print(dataA.dtype,np.max(dataA))


            """ prepare fetch_dict """
            fetch_dict = {
                "G_op": train_op["G_op"],
                "loss": scalars["loss"],
                "psnr": scalars["psnr"],
                "gstep": incre_global_step,
            }


            fetch_dict_test = {
                #"loss": scalars_test["loss"],
                "psnr": scalars_test["psnr"],
            }




            """ run """
            result = sess.run(fetch_dict, feed_dict={input_A: dataA, input_B: dataB})


            """ post_processing """
            # Print log
            if global_step.eval() % 1 == 0:
                print("Iteration %d : loss %f psrn %f" % (global_step.eval(), result["loss"], result["psnr"]))

            if global_step.eval() % 1 == 0:
                psnr = []
                elapse = []
                for i in range(7):
                    dataTest_handler = TestDataHandler(
                        # os.path.normcase('../dataset/train/BSD400'),
                        os.path.normcase('../dataset/test/Set5'),
                        2000, is_color=True)

                    start = time.time()
                    dataTest,_,_ = dataTest_handler.next()
                    dataTest_ycbcr = rgb2ycbcr_batch(dataTest)
                    dataTestB = dataTest_ycbcr[:,:,:,0:1]

                    dataTest_jpeg = cvt_jpeg(dataTest)
                    dataTest_jpeg_ycbcr = rgb2ycbcr_batch(dataTest_jpeg)
                    dataTestA = dataTest_jpeg_ycbcr[:,:,:,0:1]
                    result_test = sess.run(fetch_dict_test, feed_dict={input_TEST_A: dataTestA, input_TEST_B: dataTestB})
                    psnr.append(result_test["psnr"])
                    elapse.append(time.time() - start)
                print("Iteration %d : psnr_test %f elapse %f" % (global_step.eval(), np.mean(psnr), np.mean(elapse)))



            if global_step.eval() % 10 == 0:
                fetch_dict = {"summary": summary_op}
                result = sess.run(fetch_dict, feed_dict={input_A: dataA, input_B: dataB})
                summary_writer.add_summary(result["summary"], global_step.eval())
                summary_writer.flush()


                fetch_dict_test = {"summary": summary_op_test}
                result_test = sess.run(fetch_dict_test, feed_dict={input_TEST_A: dataTestA, input_TEST_B: dataTestB})
                summary_writer_test.add_summary(result_test["summary"], global_step.eval())
                summary_writer_test.flush()






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
    parser.add_argument("--gpu", type=int, default=0)  # -1 for CPU

    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_size", type=int, default=64)
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
    parser.add_argument("--g_type", type=int, default=2)  # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)

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
    if args.type == "YCbCr":
        args.c_dim = 1; #args.train_subdir += "_M"; args.test_subdir += "_M"
    elif args.type == "RGB":
        args.c_dim = 3;
    elif args.type == "Gray":
        args.c_dim = 1
    print("Eaxperiment tag : " + args.exp_tag)
    pp.pprint(args)
    check_folder(args.checkpoint_dir)
    check_folder(args.checkpoint_dir+"_test")
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

