import argparse
import random
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from util import loader_for_predict as ld
from util import model
from util import repoter as rp

#画像の枚数
NUM = 40

def load_dataset():
    loader = ld.LoaderForPredict(dir_original="data_set/test_images",
                       init_size=(512, 512))
    return loader.get_dataset()


def implement(parser):

    #image data
    train = load_dataset()

    #reporter
    reporter = rp.Reporter(parser=parser)

    # GPU
    gpu = parser.gpu

    #model
    model_unet = model.UNet(size=(512, 512), l2_reg=parser.l2reg).model

    # Initialize session
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    is_augment = parser.augmentation

    #Sarver
    saver = tf.train.Saver()
    if  not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    if not os.path.exists("./output"):
        os.makedirs("./output")

    if not os.path.exists("./input"):
        os.makedirs("./input")

    saver.restore(sess, "./checkpoint/save_model_512hh_epoch_210.ckpt")


    train_images_original = train.images_original
    train_images_original = train_images_original[:,:,:, np.newaxis]

    for k in range(NUM):

        print("number: ", k)
        idx_train = k
        outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train_images_original[idx_train]],
                                                    model_unet.is_training: False})
 
        images_original_size = train.images_original_size
        print(images_original_size[idx_train])

        pred_image = reporter.cast_to_out_image(outputs_train[0], images_original_size[idx_train])

        #インプット画像の確認
        image_in_np = np.squeeze(train_images_original[idx_train])
        image_in_pil = reporter.cast_to_out_image_in(image_in_np, images_original_size[idx_train])
        if k >= 40:
            #reporter.save_simple_image(pred_image, k-40, "./output", "hv_")
            #reporter.save_simple_image(image_in_pil, k-40, "./input", "hv_")
            pass
        else:
            reporter.save_simple_image(pred_image, k, "./output", "hh_")
            reporter.save_simple_image(image_in_pil, k, "./input", "hh_")

    print("Result")

    sess.close()


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.7, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    implement(parser)
