import os
from enum import Enum, auto
from typing import Optional

from organoid_tracker.config import ConfigFile, config_type_int, config_type_image_shape, config_type_float, \
    config_type_bool
from organoid_tracker.imaging import list_io


class _ChannelSpecifier:
    """Channel specifier. Supports "1" for the first channel (index 0), but also "2 from last", which becomes -2."""

    index_from_zero: int

    def __init__(self, input: str):
        if input.endswith("from end"):  # "3 from last" becomes -3
            self.index_from_zero = -int(input[:-len("from end")].strip())
        else:
            self.index_from_zero = int(input.strip()) - 1  # So "1" becomes 0

    def __str__(self):
        if self.index_from_zero >= 0:
            return str(self.index_from_zero + 1)  # So 0 becomes "1", 3 becomes "4"
        else:
            return f"{-self.index_from_zero} from end"  # So -2 becomes "2 from end"


class NetworkConfig:

    input_file: str
    output_folder: str
    starting_model_path: str
    epochs: int
    batch_size: int

    transmitted_light_channel: _ChannelSpecifier
    draw_radius_xy: float  # Used if painting_target is DIVISIONS
    draw_radius_z: float  # Used if painting_target is DIVISIONS

    patch_size_x: int
    patch_size_y: int
    patch_size_z: int

    seed: int = 1
    validation_fraction: float

    @property
    def image_volume(self) -> int:
        return self.patch_size_z * self.patch_size_y * self.patch_size_x

    @property
    def tensorboard_path(self):
        return os.path.join(self.output_folder, "tensorboard")

    @property
    def checkpoints_path(self):
        return os.path.join(self.output_folder, "checkpoints")

    @property
    def model_path(self):
        return os.path.join(self.output_folder, "saved_model")

    @property
    def examples_path(self):
        return os.path.join(self.output_folder, "examples")

    def __init__(self, config: ConfigFile):
        self.input_file = config.get_or_prompt("input_file", f"Please paste the path to the"
             f" {list_io.FILES_LIST_EXTENSION} file containing the training data")
        self.output_folder = config.get_or_default("output_folder", "training_output",
             comment="Folder that will contain all output files")
        self.starting_model_path = config.get_or_default("starting_model_path", "", comment="Specify the path to an"
             " existing model here, if you want to do transfer learning.")
        self.epochs = config.get_or_default("epochs", "1", comment="Number of epochs to train", type=config_type_int)
        self.batch_size = config.get_or_default("batch_size", "4", comment="Batch size. Lower this if you run out of"
             " memory", type=config_type_int)

        self.draw_radius_xy = config.get_or_default("draw_radius_xy", "10", comment="Draw radius in the x and y"
             " direction. This setting controls the variance in pixels of the Gaussians that are drawn at the"
             " locations of nucleus centers", type=config_type_float)
        self.draw_radius_z = config.get_or_default("draw_radius_z", "2", comment="Same as above, but in the z"
             " direction", type=config_type_float)
        self.transmitted_light_channel = config.get_or_default("transmitted_light_channel", "2", comment="Transmitted"
             " light channel, used as training input.", type=_ChannelSpecifier)
        self.patch_size_z, self.patch_size_y, self.patch_size_x = config.get_or_default("patch_size_zyx", "256,256,16",
             comment="Determines how large the training volume is", type=config_type_image_shape)
        self.max_image_z = config.get_or_default("max_image_z", "18", comment="Maximum z layer to use for training."
             " High z layers might not be useable for training, as the fluorescence signal is too dark."
             " Must not be smaller than the patch_size_z", type=config_type_int)
        self.validation_fraction = config.get_or_default("validation_fraction", "0.2", comment="Fraction of time"
             " points that is used for validation data instead of training data", type=config_type_float)
