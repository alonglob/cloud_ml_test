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


def input_fn(filename, batch_size=100):
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 3 * batch_size)

    return {'inputs': images}, labels


def get_input_fn(filename, batch_size=100):
    return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features['inputs'], [-1, 24, 24, 1])

    # Spatial Transformer
    trans = blocks.spacial_transformer(features['inputs'], input_layer, out_size=(28, 28))

    # %%
    # Build the encoder
    n_filters = [1, 10, 10, 10]
    filter_sizes = [3, 3, 3, 3]
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.relu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = tf.nn.relu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(features['inputs'])[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - input_layer))

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=cost, axis=1)
        probabilities = tf.nn.softmax(cost, name='softmax_tensor')

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
