import tensorflow as tf
from util import loader as ld

class UNet:
    def __init__(self, size=(256, 256), l2_reg=None):
        self.model = self.create_model(size, l2_reg)

    @staticmethod
    def create_model(size, l2_reg):
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 1])
        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], len(ld.DataSet.CATEGORY)])
        is_training = tf.placeholder(tf.bool)

        conv1_1 = UNet.conv(inputs, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 256, 256, 64
        conv1_2 = UNet.conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 256, 256, 64
        pool1 = UNet.pool(conv1_2)#N, 128, 128, 64

        conv2_1 = UNet.conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 128, 128, 128
        conv2_2 = UNet.conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 128, 128, 128
        pool2 = UNet.pool(conv2_2)#N, 64, 64, 128

        conv3_1 = UNet.conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 64, 64, 256
        conv3_2 = UNet.conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 64, 64, 256
        pool3 = UNet.pool(conv3_2)#N, 32, 32, 256

        conv4_1 = UNet.conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 32, 32, 512
        conv4_2 = UNet.conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)#N, 32, 32, 512
        pool4 = UNet.pool(conv4_2)#N, 16, 16, 512

        conv5_1 = UNet.conv(pool4, filters=1024, l2_reg_scale=l2_reg)#N, 16, 16, 1024
        conv5_2 = UNet.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)#N, 16, 16, 1024
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2], axis=3)
        #transpose:[N, 32, 32, 512] + conv4_2:[N, 32, 32, 512] = [N, 32, 32, 1024]

        conv_up1_1 = UNet.conv(concated1, filters=512, l2_reg_scale=l2_reg)#N, 32, 32, 512
        conv_up1_2 = UNet.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)#N, 32, 32, 512
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2], axis=3)
        #transpose:[N, 64, 64, 256] + conv3_2:[N, 64, 64, 256] = [N, 64, 64, 512]

        conv_up2_1 = UNet.conv(concated2, filters=256, l2_reg_scale=l2_reg)#N, 64, 64, 256
        conv_up2_2 = UNet.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)#N, 64, 64, 256
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2], axis=3)
        #transpose:[N,128, 128, 128] + conv2_2:[N, 128, 128, 128] = [N, 128, 128, 256]

        conv_up3_1 = UNet.conv(concated3, filters=128, l2_reg_scale=l2_reg)#N, 128, 128, 128
        conv_up3_2 = UNet.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)#N, 128, 128, 128
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2], axis=3)
        #transpose:[N, 256, 256, 64] + conv1_2:[N, 256, 256, 64] = [N, 256, 256, 128]

        conv_up4_1 = UNet.conv(concated4, filters=64, l2_reg_scale=l2_reg)#N, 256, 256, 64
        conv_up4_2 = UNet.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg)#N, 256, 256, 64
        outputs = UNet.conv(conv_up4_2, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None)
        #N, 256, 256, len(category)

        return Model(inputs, outputs, teacher, is_training)

    @staticmethod
    def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer
        )
        if batchnorm_istraining is not None:
            conved = UNet.bn(conved, batchnorm_istraining)

        return conved

    @staticmethod
    def bn(inputs, is_training):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        return normalized

    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
        return pooled

    @staticmethod
    def conv_transpose(inputs, filters, l2_reg_scale=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            strides=[2, 2],
            kernel_size=[2, 2],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        return conved


class Model:
    def __init__(self, inputs, outputs, teacher, is_training):
        self.inputs = inputs
        self.outputs = outputs
        self.teacher = teacher
        self.is_training = is_training
