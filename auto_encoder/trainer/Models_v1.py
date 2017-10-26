from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from trainer.spatial_transformer import transformer

# this model contains:
# Route 1: input - [transformer] - [conv1,output1] - [conv2] - [fc1] - [dropout] - [fc2]
# an output to tensorBoard

BN_EPSILON = 0.001

LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")


def max_pool_3x3_s1(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2_RGB(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    variable = tf.Variable(initial)
    # tf.summary.histogram(name, variable)
    return variable


def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape, name=name)
    variable = tf.Variable(initial)
    # tf.summary.histogram(name, variable)
    return variable


def spacial_transformer(x, x_image_grey, out_size=(28, 28)):
    # %% this spacial transformer uses a localisation network built of 2 Fc layers
    # %% out_size could be changed, not recommended.
    with tf.name_scope('spacial_transformer_0'):
        # %% We'll setup the two-layer localisation network to figure out the
        # %% parameters for an affine transformation of the input
        # %% Create variables for fully connected layer
        with tf.name_scope('ST_fc0'):
            W_fc_loc1 = weight_variable([784, 20], 'W_fc_loc1')
            b_fc_loc1 = bias_variable([20], 'b_fc_loc1')
            # %% Define the two layer localisation network
            h_fc_loc1 = tf.nn.relu(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
        with tf.name_scope('ST_fc1'):
            # %% We can add dropout for regularizing and to reduce overfitting like so:
            keep_prob = tf.placeholder(tf.float32)
            h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

            W_fc_loc2 = weight_variable([20, 6], 'W_fc_loc1')

            # Use identity transformation as starting point
            initial = np.array([[1., 0, 0], [0, 1., 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
            b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

            # %% Second layer
            h_fc_loc2 = tf.nn.relu(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

        # %% We'll create a spatial transformer module to identify discriminative
        # %% patches
        x_trans = transformer(x_image_grey, h_fc_loc2, out_size)

    return (x_trans, keep_prob)


def residual_module(h_conv1, number, filters, kernelSize=5):
    # %% this Residual Block contains 3 inner layers -> 1x1 -> 3x3 -> 1x1
    with tf.name_scope('residual_block' + number):


        # 1x1 convolution responsible for reducing dimension
        with tf.variable_scope('Res_conv_1x1_0' + number):
            conv = tf.layers.conv2d(
                h_conv1,
                filters=filters/2,
                kernel_size=1,
                padding='valid',
                activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)

        with tf.variable_scope('Res_conv_' + number):
            conv = tf.layers.conv2d(
                conv,
                filters=filters/2,
                kernel_size=kernelSize,
                padding='same',
                activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)

        # 1x1 convolution responsible for restoring dimension
        with tf.variable_scope('Res_conv_1x1_1' + number):
            input_dim = h_conv1.get_shape()[-1].value ##??
            conv = tf.layers.conv2d(
                conv,
                filters=filters,
                kernel_size=1,
                padding='valid',
                activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)

        # shortcut connections that turn the network into its counterpart
        # residual function (identity shortcut)
        net = conv + h_conv1

    return (net)

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    Create a variable with tf.get_variable()
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    Generate the output layer
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights0', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias0', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension, number):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta' + number, dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma' + number, dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(number, input_layer, filter_shape, stride, relu=True):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :param relu: boolean. Relu after BN?
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv'+ number, shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel, number)

    if relu is True:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer
    return output


def split(input_layer, stride, number, depth):
    '''
    The split structure in Figure 3b of the paper. It takes an input tensor. Conv it by [1, 1,
    64] filter, and then conv the result by [3, 3, 64]. Return the
    final resulted tensor, which is in shape of [batch_size, input_height, input_width, 64]

    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel/64]
    '''

    input_channel = input_layer.get_shape().as_list()[-1]
    num_filter = depth
    # according to Figure 7, they used 64 as # filters for all cifar10 task

    with tf.variable_scope('bneck_reduce_size' + number):
        conv = conv_bn_relu_layer(number=number, input_layer=input_layer, filter_shape=[1, 1, input_channel, num_filter],
                                  stride=stride)
    with tf.variable_scope('bneck_conv' + number):
        conv = conv_bn_relu_layer(number=number, input_layer=conv, filter_shape=[3, 3, num_filter, num_filter], stride=1)

    return conv


def bottleneck_b(input_layer, stride, number, cardinality, depth):
    '''
    The bottleneck strucutre in Figure 3b. Concatenates all the splits
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    split_list = []
    for i in range(cardinality):
        with tf.variable_scope('split_%i'%i + number):
            splits = split(input_layer=input_layer, stride=stride, number = number, depth=depth)
        split_list.append(splits)

    # Concatenate splits and check the dimension
    concat_bottleneck = tf.concat(values=split_list, axis=3, name='concat')

    return concat_bottleneck


def bottleneck_c1(input_layer, stride, number, cardinality, depth):
    '''
    The bottleneck strucutre in Figure 3c. Grouped convolutions
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    bottleneck_depth = depth
    with tf.variable_scope('bottleneck_c_l1' + number):
        l1 = conv_bn_relu_layer(number=number, input_layer=input_layer,
                                filter_shape=[1, 1, input_channel, bottleneck_depth],
                                stride=stride)
    with tf.variable_scope('group_conv' + number):
        filter = create_variables(name='depthwise_filter' + number, shape=[3, 3, bottleneck_depth, cardinality])
        l2 = tf.nn.depthwise_conv2d(input=l1,
                                    filter=filter,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
    return l2


def bottleneck_c(input_layer, stride, number, cardinality, depth):
    '''
    The bottleneck strucutre in Figure 3c. Grouped convolutions
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    bottleneck_depth = depth * cardinality
    with tf.variable_scope('bottleneck_c_l1' + number):
        l1 = conv_bn_relu_layer(number=number, input_layer=input_layer,
                                filter_shape=[1, 1, input_channel, bottleneck_depth],
                                stride=stride)
    with tf.variable_scope('group_conv' + number):
        filter = create_variables(name='depthwise_filter' + number, shape=[3, 3, bottleneck_depth, cardinality])
        l2 = conv_bn_relu_layer(number=number, input_layer=l1,
                                filter_shape=[3, 3, bottleneck_depth, bottleneck_depth],
                                stride=1)
    return l2


def resnext_block(input_layer, output_channel, number, cardinality=1, depth=16, b=True):
    '''
    The block structure in Figure 3b. Takes a 4D tensor as input layer and splits, concatenates
    the tensor and restores the depth. Finally adds the identity and ReLu.
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param output_channel: int, the number of channels of the output
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel do not match!!!' + str(input_channel))

    if b:
        concat_bottleneck = bottleneck_b(input_layer, stride, number, cardinality, depth)
    else:
        concat_bottleneck = bottleneck_c(input_layer, stride, number, cardinality, depth)

    bottleneck_depth = concat_bottleneck.get_shape().as_list()[-1]
    assert bottleneck_depth == depth * cardinality

    # Restore the dimension. Without relu here
    restore = conv_bn_relu_layer(number=number, input_layer=concat_bottleneck,
                                 filter_shape=[1, 1, bottleneck_depth, output_channel],
                                 stride=1, relu=False)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    # According to section 4 of the paper, relu is played after adding the identity.
    output = tf.nn.relu(restore + padded_input)

    return output

