import random
from typing import Iterable, Tuple, List, NamedTuple

import numpy
import tensorflow
from numpy import ndarray
from tensorflow.python.data import Dataset

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from ._config import NetworkConfig
from ._painting_target import create_painting_target


class _ImagePointer(NamedTuple):
    experiment: Experiment
    time_point: TimePoint


def create_datasets(experiments: List[Experiment], config: NetworkConfig) -> Tuple[Dataset, Dataset]:
    """Creates a dataset of pairs of (z, y, x, c) images: transmitted_light and nucleus_center. Dataset will already be
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
    # (for nucleus_center images, scaling from 0 to 1 is enough)
    dataset = dataset.map(lambda transmitted_light, nucleus_center: (tensorflow.image.per_image_standardization(transmitted_light), nucleus_center))

    # Data augmentation
    if train:
        dataset = dataset.map(lambda transmitted_light, nucleus_center: _perform_augmentation(transmitted_light, nucleus_center, config.seed))

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

    # Load the nucleus_center image
    nucleus_center_image = create_painting_target(config, experiment, time_point)
    if nucleus_center_image is None:
        return  # Nothing to show

    # Load transmitted light too. Here we just do basic normalization, later (after this function) we will use
    # Tensorflow code for proper normalization
    transmitted_light_channel = experiment.images.get_channels()[config.transmitted_light_channel.index_from_zero]
    transmitted_light_image = experiment.images.get_image_stack(time_point, transmitted_light_channel)
    if transmitted_light_image is None:
        return  # Cannot train on this time point
    transmitted_light_image = transmitted_light_image / numpy.max(transmitted_light_image)

    yield from _extract_standard_patches(nucleus_center_image, transmitted_light_image, config)


def _extract_standard_patches(nucleus_center_image: ndarray, transmitted_light_image: ndarray,
                              config: NetworkConfig) -> Iterable[Tuple[ndarray, ndarray]]:
    """Extracts standard patches by going through the images. Size is taken from the config, as well as whether empty
    patches should be skipped."""
    i = 0
    for x_start in range(0, nucleus_center_image.shape[2] - config.patch_size_x + 1, config.patch_size_x):
        for y_start in range(0, nucleus_center_image.shape[1] - config.patch_size_y + 1, config.patch_size_y):
            z_start_max = max(0, min(nucleus_center_image.shape[0], config.max_image_z) - config.patch_size_z)
            z_start = i % (z_start_max + 1)
            i += 1

            transmitted_light_patch = transmitted_light_image[z_start:z_start + config.patch_size_z,
                                      y_start:y_start + config.patch_size_y, x_start:x_start + config.patch_size_x,
                                      numpy.newaxis]
            nucleus_center_patch = nucleus_center_image[z_start:z_start + config.patch_size_z,
                                 y_start:y_start + config.patch_size_y, x_start:x_start + config.patch_size_x,
                                 numpy.newaxis]

            if numpy.sum(nucleus_center_patch) == 0:
                continue  # Only background, not interesting to train on

            yield transmitted_light_patch, nucleus_center_patch


@tensorflow.function
def _perform_augmentation(transmitted_light: tensorflow.Tensor, nucleus_center: tensorflow.Tensor, seed: int):
    """Just horizontal and vertical flipping for now."""
    # Unfortunately, the built-in flipping functions don't support pairs of 3D images
    # So we do it manually
    flip_hor = tensorflow.random.uniform([], seed=seed) < 0.5
    flip_vert = tensorflow.random.uniform([], seed=seed) < 0.5

    if flip_hor:
        transmitted_light = tensorflow.reverse(transmitted_light, [-1])
        nucleus_center = tensorflow.reverse(nucleus_center, [-1])

    if flip_vert:
        transmitted_light = tensorflow.reverse(transmitted_light, [-2])
        nucleus_center = tensorflow.reverse(nucleus_center, [-2])

    return transmitted_light, nucleus_center

