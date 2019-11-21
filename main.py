import argparse
import random
import tensorflow as tf
import numpy as np
import os
import easydict
#import yaml

from util import loader as ld
from util import model
from util import repoter as rp

#imageの枚数 TODO
NUM = 10
#restoreするかどうか
CONTINUE = False
#check pointをセーブするかどうか
SAVE = True

#Restoreするモデルのファイル名
RESTORE_MODEL = "save_model_done.ckpt"


def load_dataset(train_rate):
    loader = ld.Loader(dir_original="data_set/train_images",
                       dir_segmented="data_set/train_annotations",
                       init_size=(128, 128))
    return loader.load_train_test(train_rate=train_rate, shuffle=False)


def train(parser):

    train, test = load_dataset(train_rate=parser.trainrate)
    valid = test.devide(0, int(NUM*0.1))
    test = test.devide(int(NUM*0.1), int(NUM*0.2))

    #保存ファイル
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    #GPU
    gpu = parser.gpu

    #model
    model_unet = model.UNet(size=(128, 128), l2_reg=parser.l2reg).model

    #誤差関数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    #精度
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #gpu config
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #parameter
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation


    v_images_original = valid.images_original
    v_images_original = v_images_original[:,:,:, np.newaxis]
    t_images_original = test.images_original
    t_images_original = t_images_original[:,:,:, np.newaxis]

    train_dict = {model_unet.inputs: v_images_original, model_unet.teacher: valid.images_segmented,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: t_images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}


    saver = tf.train.Saver()
    if  not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    if CONTINUE:
        if os.path.exists("./checkpoint/"+ RESTORE_MODEL):
            saver.restore(sess, "./checkpoint/"+ RESTORE_MODEL)

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size):#ここでtrainがシャッフルされる
            # バッチデータ
            images_original = batch.images_original
            if not is_augment:
                images_original = images_original[:, :, :, np.newaxis]
            inputs = images_original
            teacher = batch.images_segmented

            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})

        #入力データはグレースケールなのでチャネルの分の次元を追加
        train_images_original = train.images_original
        train_images_original = train_images_original[:,:,:, np.newaxis]

        # 評価
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 1 == 0:
                idx_train = random.randrange(NUM*0.8)#trainサイズ
                idx_test = random.randrange(NUM*0.1)#validationとtest
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train_images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [t_images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train_images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]] #なぜかtrain.images_segmentedがシャッフルされるTODO
                test_set = [t_images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=len(ld.DataSet.CATEGORY)-1)
        if epoch % 10 == 0:
            if SAVE:
                save_path = saver.save(sess, "./checkpoint/save_model_epoch_"+str(epoch)+"_.ckpt")
    save_path = saver.save(sess, "./checkpoint/save_model_done_plus.ckpt")

    #modelの評価
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    sess.close()


def get_parser():
    args = easydict.EasyDict({
        "batchsize": 2,
        "epoch": 300,
        "gpu": True,
        "augmentation":False,
        "l2reg":0.0001,
        "trainrate":0.8
    })
    return args #parser.parse_args()


if __name__ == '__main__':
    parser = get_parser()#.parse_args()
    train(parser)
