"This code is partially derived from the following websites and articles:"
"https://github.com/zhanghang1989/ResNeSt"
"""""@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}"""""
"https://github.com/QiaoranC/tf_ResNeSt_RegNet_model"
"https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/"


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import tensorflow
import datetime as dt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    UpSampling2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import pandas as pd




## Initialising image size and channels
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)
Image_Channels=3


## Setting up the directory and labels. Please note this directory might differ
## So, please adjust directory based on your computer.

## This part of code is not needed of you are using metadata
# dir = "./Dataset/Dataset"
# labels = ['fake', 'real']
# categories=[]
# filename = []
# for label in labels:
#     for f_name in os.listdir(dir + '/' + label + '/'):
#         category = label
#         filename.append(f_name)
#         if category=='fake':
#             categories.append(1)
#         else:
#             categories.append(0)
# df=pd.DataFrame({
#     'filename':filename,
#     'category':categories
# })

## Reading and extracting Metadata
df = pd.read_csv("df.csv")
filename = df['filename']
category = df['category']

##Converting Labels and Defining Batch Size
df["category"] = df["category"].replace({0:'real',1:'fake'})
train_df,testvalidate_df = train_test_split(df,test_size=0.30,
  random_state=12)
train_df = train_df.reset_index(drop=True)
testvalidate_df = testvalidate_df.reset_index(drop=True)
validate_df,test_df = train_test_split(testvalidate_df,test_size=0.50,
  random_state=12)
test_df = test_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
total_test=test_df.shape[0]
batch_size= 32

## Generating Training set, Validation Set, and Testing Set
train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 "./Dataset/Dataset/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "./Dataset/Dataset/",
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                 "./Dataset/Dataset/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 shuffle=False)


##### RESNEST MODEL #####

## Defining Functions and classes to implement RESNEST Architecture:

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(
        graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd="op", options=opts
    )

    return flops.total_float_ops  # Prints the "flops" of the model.


class Mish(Activation):
    """
    based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    Mish Activation Function.
    """

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    # with tf.device("CPU:0"):
    result = inputs * tf.math.tanh(tf.math.softplus(inputs))
    return result


class GroupedConv2D(object):
    """Groupped convolution.
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_py
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel size.
    """

    def __init__(self, filters, kernel_size, use_keras=True, **kwargs):
        """Initialize the layer.
        Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel for each group.
        use_keras: An boolean value, whether to use keras layer.
        **kwargs: other parameters passed to the original conv2d layer.
        """
        self._groups = len(kernel_size)
        self._channel_axis = -1

        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(self._get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs))

    def _get_conv2d(self, filters, kernel_size, use_keras, **kwargs):
        """A helper function to create Conv2D layer."""
        if use_keras:
            return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        if tf.__version__ < "2.0.0":
            filters = inputs.shape[self._channel_axis].value
        else:
            filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x


class ResNest:
    def __init__(self, verbose=False, input_shape=(128, 128, 3), active="relu", n_classes=2,
                 dropout_rate=0.2, fc_activation=None, blocks_set=[3, 4, 6, 3], radix=2, groups=1,
                 bottleneck_width=64, deep_stem=True, stem_width=32, block_expansion=4, avg_down=True,
                 avd=True, avd_first=False, preact=False, using_basic_block=False, using_cb=False):
        self.channel_axis = -1  # not for change
        self.verbose = verbose
        self.active = active  # default relu
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        self.blocks_set = blocks_set
        self.radix = radix
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.block_expansion = block_expansion
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        # self.cardinality = 1
        self.dilation = 1
        self.preact = preact
        self.using_basic_block = using_basic_block
        self.using_cb = using_cb

    def _make_stem(self, input_tensor, stem_width=64, deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = Conv2D(stem_width, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width * 2, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)

            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        else:
            x = Conv2D(stem_width, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)
            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        return x

    def _rsoftmax(self, input_tensor, filters, radix, groups):
        x = input_tensor
        batch = x.shape[0]
        if radix > 1:
            x = tf.reshape(x, [-1, groups, radix, filters // groups])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.keras.activations.softmax(x, axis=1)
            x = tf.reshape(x, [-1, 1, 1, radix * filters])
        else:
            x = Activation("sigmoid")(x)
        return x

    def _SplAtConv2d(self, input_tensor, filters=64, kernel_size=3, stride=1, dilation=1, groups=1, radix=0):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        x = GroupedConv2D(filters=filters * radix, kernel_size=[kernel_size for i in range(groups * radix)],
                          use_keras=True, padding="same", kernel_initializer="he_normal", use_bias=False,
                          data_format="channels_last", dilation_rate=dilation)(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        batch, rchannel = x.shape[0], x.shape[-1]
        if radix > 1:
            splited = tf.split(x, radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        # print('sum',gap.shape)
        gap = GlobalAveragePooling2D(data_format="channels_last")(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])
        # print('adaptive_avg_pool2d',gap.shape)

        reduction_factor = 4
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        x = Conv2D(inter_channels, kernel_size=1)(gap)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters * radix, kernel_size=1)(x)

        atten = self._rsoftmax(x, filters, radix, groups)

        if radix > 1:
            logits = tf.split(atten, radix, axis=-1)
            out = sum([a * b for a, b in zip(splited, logits)])
        else:
            out = atten * x
        return out

    def _make_block(
            self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False,
            is_first=False
    ):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            short_cut = input_tensor
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same",
                                                 data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(
                        short_cut)
                short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same",
                                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(
                    short_cut)
            else:
                short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=stride, padding="same",
                                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(
                    short_cut)

            short_cut = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(short_cut)
        else:
            short_cut = input_tensor

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
        x = Conv2D(group_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                   use_bias=False,
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                  groups=self.cardinality, radix=radix)
        else:
            x = Conv2D(group_width, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')
        x = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same",
                   kernel_initializer="he_normal",
                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

        m2 = Add()([x, short_cut])
        m2 = Activation(self.active)(m2)
        return m2

    def _make_block_basic(
            self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False,
            is_first=False
    ):
        """Conv2d_BN_Relu->Bn_Relu_Conv2d
        """
        x = input_tensor
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        short_cut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same",
                                                 data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(
                        short_cut)
                short_cut = Conv2D(filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                                   use_bias=False, data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(filters, kernel_size=1, strides=stride, padding="same",
                                   kernel_initializer="he_normal",
                                   use_bias=False, data_format="channels_last")(short_cut)

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                  groups=self.cardinality, radix=radix)
        else:
            x = Conv2D(filters, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        m2 = Add()([x, short_cut])
        return m2

    def _make_layer(self, input_tensor, blocks=4, filters=64, stride=2, is_first=True):
        x = input_tensor
        if self.using_basic_block is True:
            x = self._make_block_basic(x, first_block=True, filters=filters, stride=stride, radix=self.radix,
                                       avd=self.avd, avd_first=self.avd_first, is_first=is_first)
            # print('0',x.shape)

            for i in range(1, blocks):
                x = self._make_block_basic(
                    x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd,
                    avd_first=self.avd_first
                )
                # print(i,x.shape)

        elif self.using_basic_block is False:
            x = self._make_block(x, first_block=True, filters=filters, stride=stride, radix=self.radix, avd=self.avd,
                                 avd_first=self.avd_first, is_first=is_first)
            # print('0',x.shape)

            for i in range(1, blocks):
                x = self._make_block(
                    x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd,
                    avd_first=self.avd_first
                )
                # print(i,x.shape)
        return x

    def _make_Composite_layer(self, input_tensor, filters=256, kernel_size=1, stride=1, upsample=True):
        x = input_tensor
        x = Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        if upsample:
            x = UpSampling2D(size=2)(x)
        return x

    def build(self):
        get_custom_objects().update({'mish': Mish(mish)})

        input_sig = Input(shape=self.input_shape)
        x = self._make_stem(input_sig, stem_width=self.stem_width, deep_stem=self.deep_stem)

        if self.preact is False:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
        if self.verbose:
            print("stem_out", x.shape)

        x = MaxPool2D(pool_size=3, strides=2, padding="same", data_format="channels_last")(x)
        if self.verbose:
            print("MaxPool2D out", x.shape)

        if self.preact is True:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if self.using_cb:
            second_x = x
            second_x = self._make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
            second_x_tmp = self._make_Composite_layer(second_x, filters=x.shape[-1], upsample=False)
            if self.verbose: print('layer 0 db_com', second_x_tmp.shape)
            x = Add()([second_x_tmp, x])
        x = self._make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
        if self.verbose:
            print("-" * 5, "layer 0 out", x.shape, "-" * 5)

        b1_b3_filters = [64, 128, 256, 512]
        for i in range(3):
            idx = i + 1
            if self.using_cb:
                second_x = self._make_layer(x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
                second_x_tmp = self._make_Composite_layer(second_x, filters=x.shape[-1])
                if self.verbose: print('layer {} db_com out {}'.format(idx, second_x_tmp.shape))
                x = Add()([second_x_tmp, x])
            x = self._make_layer(x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
            if self.verbose: print('----- layer {} out {} -----'.format(idx, x.shape))

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if self.verbose:
            print("pool_out:", x.shape)  # remove the concats var

        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, noise_shape=None)(x)

        fc_out = Dense(self.n_classes, kernel_initializer="he_normal", use_bias=False, name="fc_NObias")(
            x)  # replace concats to x
        if self.verbose:
            print("fc_out:", fc_out.shape)

        if self.fc_activation:
            fc_out = Activation(self.fc_activation)(fc_out)

        model = models.Model(inputs=input_sig, outputs=fc_out)

        if self.verbose:
            print("Resnest builded with input {}, output{}".format(input_sig.shape, fc_out.shape))
            print("-------------------------------------------")
            print("")

        return model


def get_model(model_name='ResNest50', input_shape=(128, 128, 3), n_classes=2,
              verbose=False, dropout_rate=0, fc_activation=None, **kwargs):

    model_name = model_name.lower()

    resnest_parameters = {
        'resnest50': {
            'blocks_set': [3, 4, 6, 3],
            'stem_width': 224,
        },
        'resnest101': {
            'blocks_set': [3, 4, 23, 3],
            'stem_width': 64,
        },
        'resnest200': {
            'blocks_set': [3, 24, 36, 3],
            'stem_width': 64,
        },
        'resnest269': {
            'blocks_set': [3, 30, 48, 8],
            'stem_width': 64,
        },
    }

    if model_name in resnest_parameters.keys():
        model = ResNest(verbose=verbose, input_shape=input_shape,
                        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
                        blocks_set=resnest_parameters[model_name]['blocks_set'], radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True,
                        stem_width=resnest_parameters[model_name]['stem_width'], avg_down=True, avd=True,
                        avd_first=False, **kwargs).build()

    else:
        raise ValueError('Unrecognize model name {}'.format(model_name))
    return model

if __name__ == "__main__":

    model_names = ['ResNest50']
    input_shape = [128, 128, 3]
    n_classes = 2
    fc_activation = 'softmax'  # softmax sigmoid

    for model_name in model_names:
        print('model_name', model_name)
        model = get_model(model_name=model_name, input_shape=input_shape, n_classes=n_classes,
                          verbose=True, fc_activation=fc_activation)
        print('-' * 10)

## Defining Early Stopping

earlystop = EarlyStopping(patience = 50)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 50,verbose = 1,factor = 0.5,min_lr = 0.01)
callbacks = [earlystop,learning_rate_reduction]

## Getting model Information
model_name = 'ResNest50'
input_shape = [128,128,3]
n_classes = 2
fc_activation = 'softmax'
active = 'relu' # relu or mish

model = get_model(model_name=model_name,
                  input_shape=input_shape,
                  n_classes=n_classes,
                  fc_activation=fc_activation,
                  active=active,
                  verbose=False,
                 )

## Model Compile
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['acc'])

## Model Training
import time
epochs=100
t1 = time.time()
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks, verbose=1)
tot_time = time.time() - t1
print('Trainig took %s seconds' %round(tot_time))

## Saving the model
model.save("resnest.h5")


## Plotting the tarining history
def summarize_diagnostics(history):
    # plot loss
    pyplot.figure(figsize=(10,8))
    pyplot.subplot(212)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='val')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(211)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='val')
    pyplot.legend()
    filename = sys.argv[0].split('/')[-1]
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.49)
    pyplot.show(filename)
    pyplot.close()

summarize_diagnostics(history)


## Make predictions based on the testing dataset
nb_samples = test_df.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size), verbose=1)
## Calculation of the testing dataset accuracy
_, acc = model.evaluate_generator(test_generator,steps=np.ceil(nb_samples/batch_size), verbose=1)
print('> %.3f' % (acc * 100.0))

## Confusion Matrix calculations and plotting
y_true = test_generator.classes
y_pred = predict > 0.5
y_pred = list(map(int, y_pred[:,1]))
font = {
'family': 'Times New Roman',
'size': 12
}
matplotlib.rc('font', **font)
mat = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(mat, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size
plt.show()

## Plotting ROC Curve and AU-ROC calculation for RENSEST
auc = roc_auc_score(y_true, predict[:,1])
print('ResNeSt ROC AUC=%.3f' % (auc))
fpr, tpr, _ = roc_curve(y_true, predict[:,1])
pyplot.plot(fpr, tpr, marker='.', label='ResNeSt')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
