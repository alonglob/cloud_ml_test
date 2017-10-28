# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import trainer.Models_v1 as blocks
import gzip, os
import cPickle as pickle

tf.logging.set_verbosity(tf.logging.INFO)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def input_fn(filename, batch_size=1):
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, image], batch_size=batch_size,
        capacity=1)

    return {'inputs': images}, images


def get_input_fn(filename, batch_size=1):
    return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features['inputs'], [-1, 28, 28, 1])

    # Spatial Transformer
    #trans = blocks.spacial_transformer(features['inputs'], input_layer, out_size=(28, 28))

    ### Encoder
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 28x28x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 14x14x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 14x14x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 7x7x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 7x7x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 4x4x16

    ### Decoder
    upsample1 = tf.image.resize_images(encoded, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 7x7x16
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 7x7x16
    upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 14x14x16
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 14x14x32
    upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 28x28x32
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 28x28x32

    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)
    # Now 28x28x1

    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)

    # Pass logits through sigmoid and calculate the Huber loss
    label_indices = tf.cast(labels, tf.int32)
    onehot_labels = tf.one_hot(label_indices[1], depth=784)
    loss = tf.losses.huber_loss(labels=onehot_labels, predictions=logits)

    # Get cost
    cost = tf.reduce_mean(loss)

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=cost, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=cost, eval_metric_ops=eval_metric_ops)


def build_estimator(model_dir):
    return tf.estimator.Estimator(
        model_fn=_cnn_model_fn,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_steps=50))


def serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
