import random
from typing import Iterable, Tuple, List, NamedTuple

import numpy
import tensorflow
from numpy import ndarray
from tensorflow.python.data import Dataset

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from ._config import NetworkConfig


class _ImagePointer(NamedTuple):
    experiment: Experiment
    time_point: TimePoint


def create_datasets(experiments: List[Experiment], config: NetworkConfig) -> Tuple[Dataset, Dataset]:
    """Creates a dataset of pairs of (z, y, x, c) images: transmitted_light and fluorescence. Dataset will already be
    shuffled."""
    image_pointers = list(_gather_time_points(experiments))
    random.Random(config.seed).shuffle(image_pointers)

    train_images = image_pointers[int(config.validation_fraction * len(image_pointers)):]
    validation_images = image_pointers[0:int(config.validation_fraction * len(image_pointers))]
    print(f"We have {len(train_images)} train time points and {len(validation_images)} validation time points.")

    return _to_tensorflow_dataset(train_images, config, train=True), _to_tensorflow_dataset(validation_images, config, train=False)


def _to_tensorflow_dataset(image_pointers: List[_ImagePointer], config: NetworkConfig, *, train: bool) -> Dataset:
    def gen():
        for time_point_and_experiment in image_pointers:
            yield from _load_training_images_from_time_point(time_point_and_experiment, config)

    dataset = Dataset.from_generator(gen, output_signature=(
        tensorflow.TensorSpec(shape=(config.patch_size_z, config.patch_size_y, config.patch_size_x, 1),
                              dtype=tensorflow.float32),
        tensorflow.TensorSpec(shape=(config.patch_size_z, config.patch_size_y, config.patch_size_x, 1),
                              dtype=tensorflow.float32)))

    # Standardize brightness and contrast of transmitted light
    # (for fluorescence images, scaling from 0 to 1 is enough)
    dataset = dataset.map(lambda transmitted_light, fluorescence: (tensorflow.image.per_image_standardization(transmitted_light), fluorescence))

    # Data augmentation
    if train:
        dataset = dataset.map(lambda transmitted_light, fluorescence: _perform_augmentation(transmitted_light, fluorescence, config.seed))

    # Prefetching, speeds things up a lot
    dataset = dataset.prefetch(10)

    # The input list is already shuffled. This additional shuffling ensures that one single batch contains images from
    # different time points, and that each iteration is somewhat different
    dataset = dataset.shuffle(100, seed=config.seed)
    return dataset


def _gather_time_points(experiments: List[Experiment]) -> Iterable[_ImagePointer]:
    for experiment in experiments:
        for time_point in experiment.images.time_points():
            yield _ImagePointer(experiment=experiment, time_point=time_point)


def _load_training_images_from_time_point(time_point_in_experiment: _ImagePointer, config: NetworkConfig) -> Iterable[Tuple[ndarray, ndarray]]:
    experiment = time_point_in_experiment.experiment
    time_point = time_point_in_experiment.time_point

    # Load the fluorescence image
    fluorescence_channel = experiment.images.get_channels()[config.fluorescence_channel.index_from_zero]
    fluorescence_image = experiment.images.get_image_stack(time_point, fluorescence_channel)
    if fluorescence_image is None:
        return  # Nothing to show

    # Normalize that image from 0 to 1
    max_value = numpy.max(fluorescence_image)
    if max_value == 0:
        return  # No fluorescent signal in this image, ignore it
    fluorescence_image = fluorescence_image / max_value
    fluorescence_image[fluorescence_image > 1] = 1

    # Load transmitted light too. Here we just do basic normalization, later (after this function) we will use
    # Tensorflow code for proper normalization
    transmitted_light_channel = experiment.images.get_channels()[config.transmitted_light_channel.index_from_zero]
    transmitted_light_image = experiment.images.get_image_stack(time_point, transmitted_light_channel)
    if transmitted_light_image is None:
        return  # Cannot train on this time point
    transmitted_light_image = transmitted_light_image / numpy.max(transmitted_light_image)

    yield from _extract_standard_patches(fluorescence_image, transmitted_light_image, config)


def _extract_standard_patches(fluorescence_image: ndarray, transmitted_light_image: ndarray,
                              config: NetworkConfig) -> Iterable[Tuple[ndarray, ndarray]]:
    """Extracts standard patches by going through the images. Size is taken from the config, as well as whether empty
    patches should be skipped."""
    i = 0
    for x_start in range(0, fluorescence_image.shape[2] - config.patch_size_x + 1, config.patch_size_x):
        for y_start in range(0, fluorescence_image.shape[1] - config.patch_size_y + 1, config.patch_size_y):

            # Randomize the z at which we start. In this way, if we have a shape of 35 and a z_size of 10, then we will
            # sometimes get [0-9], [10-19] and [20-29], but also sometimes [6-15], [16-25] and [26-35].
            z_offset_max = max(0, fluorescence_image.shape[0] % config.patch_size_z)
            z_offset = i % (z_offset_max + 1)
            i += 1

            for z_start in range(z_offset, fluorescence_image.shape[0] - config.patch_size_z + 1, config.patch_size_z):
                transmitted_light_patch = transmitted_light_image[z_start:z_start + config.patch_size_z,
                                          y_start:y_start + config.patch_size_y, x_start:x_start + config.patch_size_x,
                                          numpy.newaxis]
                fluorescence_patch = fluorescence_image[z_start:z_start + config.patch_size_z,
                                     y_start:y_start + config.patch_size_y, x_start:x_start + config.patch_size_x,
                                     numpy.newaxis]

                if numpy.max(fluorescence_patch) < 0.5:
                    # Only background, not interesting to train on (except in rare cases)
                    if config.leak_empty_fraction > 0 and i % int(1 / config.leak_empty_fraction) != 0:
                        continue

                yield transmitted_light_patch, fluorescence_patch

@tensorflow.function
def _perform_augmentation(transmitted_light: tensorflow.Tensor, fluorescence: tensorflow.Tensor, seed: int):
    """Just horizontal and vertical flipping for now."""
    # Unfortunately, the built-in flipping functions don't support pairs of 3D images
    # So we do it manually
    flip_hor = tensorflow.random.uniform([], seed=seed) < 0.5
    flip_vert = tensorflow.random.uniform([], seed=seed) < 0.5

    if flip_hor:
        transmitted_light = tensorflow.reverse(transmitted_light, [-1])
        fluorescence = tensorflow.reverse(fluorescence, [-1])

    if flip_vert:
        transmitted_light = tensorflow.reverse(transmitted_light, [-2])
        fluorescence = tensorflow.reverse(fluorescence, [-2])

    return transmitted_light, fluorescence

