import json
import os

import tensorflow as tf

from ._config import NetworkConfig
from ._loss_functions import weighted_loss, position_recall, position_precision, overcount


class _NormalizeImages(tf.keras.layers.Layer):
    """Used to normalize an image such that it has mean 0 and stdev 1. Expects 3D images."""

    def call(self, images: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.image.per_image_standardization(images)


class _MoveZAxisLast(tf.keras.layers.Layer):
    """Converts batch,z,y,x to batch,y,x,z"""

    def call(self, images: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.transpose(images, [0, 2, 3, 1])


class _MoveZAxisBackAndAddChannel(tf.keras.layers.Layer):
    """Converts  batch,y,x,z back to batch,z,y,x"""

    def call(self, images: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.expand_dims(tf.transpose(images, [0, 3, 1, 2]), -1)


def build_model(config: NetworkConfig) -> tf.keras.Model:
    # Input layer
    input = tf.keras.Input(shape=(None, None, None, 1))
    layer = input

    # convolutions
    to_concat = []

    filter_sizes = [3, 16, 64, 128, 256]
    n=2
    layer, to_concat_layer = _conv_block(n, layer, filters=filter_sizes[1], kernel=(1, 3, 3), pool_size=(1, 2, 2),
                                         pool_strides=(1, 2, 2), name="down1")
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = _conv_block(n, layer, filters=filter_sizes[2], name="down2")
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = _conv_block(n, layer, filters=filter_sizes[3], name="down3")
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = _conv_block(n, layer, filters=filter_sizes[4], name="down4")
    to_concat.append(to_concat_layer)

    layer = _deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[4], name="up1")
    layer = _deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[3], name="up2")
    layer = _deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[2], name="up3")
    layer = _deconv_block(2, layer, to_concat.pop(), filters=filter_sizes[1], kernel=(1, 3, 3), strides=(1, 2, 2), dropout=False, name="up4")

    # apply final batch_normalization
    layer = tf.keras.layers.BatchNormalization()(layer)

    output = tf.keras.layers.Conv3D(filters=1, kernel_size=3, padding="same", activation='relu', name='out_conv')(layer)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer='Adam', loss=weighted_loss, metrics=[position_recall, position_precision, overcount])

    return model


def _conv_block(n_conv, layer, filters, kernel=(3, 3, 3), pool_size=(2, 2, 2), pool_strides=(2, 2, 2), dropout=False, name=None):
    for index in range(n_conv):
        layer = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                       name=name + '/conv{0}'.format(index + 1))(
            layer)  # To test : is coordconv needed in all layers or just first?

        if dropout:
            layer = tf.keras.layers.SpatialDropout3D(rate=0.5)(layer)

    to_concat = layer
    layer = tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_strides, padding='same',
                                         name=name + '/pool')(layer)

    layer = tf.keras.layers.BatchNormalization()(layer)

    return layer, to_concat


def _deconv_block(n_conv, layer, to_concat, filters, kernel=(3, 3, 3), strides=(2, 2, 2), dropout=False, name=None):
    layer = tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding='same',
                                            name=name + '/upconv')(layer)

    for index in range(n_conv):
        layer = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                       name=name + '/conv{0}'.format(index + 1))(layer)

        if dropout:
            layer = tf.keras.layers.SpatialDropout3D(rate=0.5)(layer)

    layer = tf.concat([layer, to_concat], axis=-1)

    layer = tf.keras.layers.BatchNormalization()(layer)

    return layer




def save_training_settings(network_config: NetworkConfig):
    """Saves the settings.json file to the model folder."""
    data = {
        "type": "cell_painting",
        "patch_size_xyz": [network_config.patch_size_x, network_config.patch_size_y, network_config.patch_size_z]
    }
    with open(os.path.join(network_config.model_path, "settings.json"), "w") as handle:
        json.dump(data, handle)
