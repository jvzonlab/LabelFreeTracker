import os
from typing import List, Dict, Any

import numpy
import tensorflow
import tifffile
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import LambdaCallback

from organoid_tracker.config import ConfigFile
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers
from . import _network, _dataset
from ._config import NetworkConfig


def train(args: List[str]) -> int:
    import tensorflow as tf

    config_file = ConfigFile("train_nucleus_centers_from_transmitted_light")
    network_config = NetworkConfig(config_file)
    config_file.save_and_exit_if_changed()

    experiments = list(list_io.load_experiment_list_file(network_config.input_file))
    model = _network.build_model(network_config)
    train_set, validation_set = _dataset.create_datasets(experiments, network_config)

    if network_config.starting_model_path:
        model.load_weights(network_config.starting_model_path)

    # for brightfield, fluorescence in train_set.as_numpy_iterator():
    #     brightfield = brightfield[:, :, :, 0]
    #     fluorescence = fluorescence[:, :, :, 0]
    #     tifffile.imshow(brightfield, cmap="gray", title="Transmitted light")
    #     plt.show()
    #     tifffile.imshow(fluorescence, cmap="gray", title="Fluorescence")
    #     plt.show()


    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=network_config.tensorboard_path,
        update_freq=100
    )
    examples_callback = _create_images_callback(model, network_config, validation_set)

    model.summary()
    history = model.fit(
        train_set.batch(network_config.batch_size),
        epochs=network_config.epochs,
        validation_data=validation_set.batch(network_config.batch_size),
        callbacks=[tensorboard_callback, examples_callback]
    )

    model.save(network_config.model_path)
    _network.save_training_settings(network_config)
    return 0


def _create_images_callback(model: tensorflow.keras.Model, config: NetworkConfig, validation_set: Dataset
                            ) -> LambdaCallback:
    """Returns a callback that will write an example image every epoch."""
    example_images = list()
    validation_set_iterator = iter(validation_set.as_numpy_iterator())
    for i in range(10):
        example_image, ground_truth = next(validation_set_iterator)
        example_images.append(example_image)
        untrained_output = model.predict(example_image[numpy.newaxis, ...])[0]

        os.makedirs(os.path.join(config.examples_path, f"Example {i+1}"), exist_ok=True)
        tifffile.imwrite(os.path.join(config.examples_path, f"Example {i+1}", "input.tif"), example_image, compress=9)
        tifffile.imwrite(os.path.join(config.examples_path, f"Example {i+1}", "ground_truth.tif"), ground_truth, compress=9)
        tifffile.imwrite(os.path.join(config.examples_path, f"Example {i+1}", "output-before-training.tif"), untrained_output, compress=9)
    del ground_truth, untrained_output  # Saves some memory

    def _write_example_image(epoch: int, logs: Dict[str, Any]):
        for i, example_image in enumerate(example_images):
            output = model.predict(example_image[numpy.newaxis, ...])[0]
            tifffile.imwrite(os.path.join(config.examples_path, f"Example {i+1}", f"output-end-of-epoch-{epoch + 1}.tif"), output, compress=9)

    return LambdaCallback(on_epoch_end=_write_example_image)
