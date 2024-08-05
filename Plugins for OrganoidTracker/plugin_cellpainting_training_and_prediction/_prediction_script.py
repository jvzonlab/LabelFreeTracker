"""A script (called from the main file) that uses the network to paint cells."""
import json
import math
import os
from typing import List

import numpy
import tifffile
from numpy import ndarray

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
import tensorflow as tf

from organoid_tracker.imaging import list_io
from . import _network


Z_DIVISIBLE_BY = 8  # Network only accepts multiples of this number as input z size
Z_MAX_SIZE = 32  # Must be a multiple of Z_DIVISIBLE_BY


def predict(args: List[str]) -> int:
    config = ConfigFile("predict_cell_painting")
    input_file = config.get_or_prompt("input_file", f"Please paste the path to the {list_io.FILES_LIST_EXTENSION} file"
                                                    f" containing the prediction data.")
    transmitted_light_channel = config.get_or_default("transmitted_light_channel", "2", comment="In what channel are the"
                                          " transmitted light images?", type=config_type_int)
    model_path = config.get_or_prompt("model_path", "What is the path to the model folder? This folder must contain a"
                                                    " settings.json file.")
    output_folder_pattern = config.get_or_default("output_folder", "output_images_{i}_{name}",
            comment="The path to the output images folder. {i} will be replaced by an arbitrary number, {name} by the"
                    " experiment name.")
    config.save()

    # Parse images
    experiments = list_io.load_experiment_list_file(input_file)

    # Parse model metadata
    if not os.path.exists(os.path.join(model_path, "settings.json")):
        print("No settings.json file in model folder")
        return 1
    with open(os.path.join(model_path, "settings.json")) as handle:
        metadata = json.load(handle)
    if metadata["type"] != "cell_painting":
        print(f"This is a {metadata['type']} model, not a cell_painting model.")
        return 1

    # Run!
    for i, experiment in enumerate(experiments):
        output_folder = output_folder_pattern.format(i=i+1, name=experiment.name.get_save_name())
        os.makedirs(output_folder, exist_ok=True)
        channel = experiment.images.get_channels()[transmitted_light_channel - 1]
        model = tf.keras.models.load_model(model_path)
        for time_point in experiment.images.time_points():
            print(f"Working on {experiment.name}, time point {time_point.time_point_number()}...")
            image = experiment.images.get_image_stack(time_point, channel)
            if image is None:
                continue
            image = image / numpy.max(image)
            image = tf.image.per_image_standardization(image).numpy()

            z_size_model = metadata["patch_size_xyz"][2]
            if z_size_model <= 8:
                output_image = _predict_in_small_nonoverlapping_parts(image, model, z_size_model)
            else:
                output_image = _predict_in_parts(image, model)
            output_image = output_image * 255
            numpy.clip(output_image, 0, 255, out=output_image)
            output_image = output_image.astype(numpy.uint8)
            tifffile.imwrite(os.path.join(output_folder, f"cell_painting_t{time_point.time_point_number():03}.tif"), output_image, 
                             compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

    print("Done!")
    return 0


def _predict_in_small_nonoverlapping_parts(image: ndarray, model: tf.keras.Model, z_size: int) -> ndarray:
    output_image = numpy.empty_like(image)
    for z in range(0, image.shape[0], z_size):

        if z + z_size > image.shape[0]:
            # Make sure that last prediction also has the correct size
            z = image.shape[0] - z_size

        output_image[z:z + z_size] = _predict_direct(model, image[z:z + z_size])
    return output_image


def _predict_in_parts(image: ndarray, model: tf.keras.Model) -> ndarray:
    """Calls the predict function, if necessary multiple times, so that the entire image is predicted."""
    if image.shape[0] <= Z_MAX_SIZE:
        desired_image_z_size = int(math.ceil(image.shape[0] / Z_DIVISIBLE_BY) * Z_DIVISIBLE_BY)
        if image.shape[0] <= desired_image_z_size:
            # Expand the image by adding black pixels
            larger_image = numpy.zeros_like(image, shape=(desired_image_z_size, image.shape[1], image.shape[2]))
            larger_image[0:image.shape[0]] = image
            return _predict_direct(model, larger_image)[0: image.shape[0]]

    # Divide the image up into parts
    output_image = numpy.empty_like(image)
    z_starts = list(range(0, image.shape[0], Z_MAX_SIZE - 4))  # The -4 ensures some overlap between images
    z_starts.reverse()  # Make sure output of lower z (closer the objective) overwrites others
    if z_starts[0] + Z_MAX_SIZE > image.shape[0]:  # Make sure first entry doesn't reach outside the image
        z_starts[0] = image.shape[0] - Z_MAX_SIZE
    for z_start in z_starts:
        output_image[z_start:z_start + Z_MAX_SIZE] = _predict_direct(model, image[z_start:z_start + Z_MAX_SIZE])
    return output_image


def _predict_direct(model: tf.keras.Model, image: ndarray):
    """Adds (unused) batch and channel indices, calls model.predict, then removes those extra indices again."""
    # Add batch and channels array
    input_array = numpy.expand_dims(image, -1)
    input_array = numpy.expand_dims(input_array, 0)
    output_image = model.predict(input_array)
    # Save as 3D 8-bits image
    output_image = output_image[0, :, :, :, 0]
    return output_image
