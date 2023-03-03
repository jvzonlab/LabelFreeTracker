from typing import Optional, Tuple, Iterable, List

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.gaussian import Gaussian
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import image_adder
from ._config import NetworkConfig


_CACHED_GAUSSIAN: Optional[Gaussian] = None
_CACHED_GAUSSIAN_ARRAY: Optional[Image] = None


def _draw_gaussian(image: ndarray, image_position: Position, draw_radius_xy: float, draw_radius_z: float):
    global _CACHED_GAUSSIAN, _CACHED_GAUSSIAN_ARRAY  # We don't want to redraw the same array over and over

    gaussian = Gaussian(1, 0, 0, 0, draw_radius_xy ** 2, draw_radius_xy ** 2, draw_radius_z ** 2, 0, 0, 0)
    if _CACHED_GAUSSIAN is None or not gaussian.almost_equal(_CACHED_GAUSSIAN):
        # Gaussian changed, make sure to wipe the cached array
        _CACHED_GAUSSIAN = gaussian
        bounds = gaussian.get_bounds()  # Will be something like "from x = -10 to x = 10"
        gaussian_stamp = numpy.zeros(bounds.get_size_zyx(), dtype=numpy.float32)
        gaussian.translated(-bounds.min_x, -bounds.min_y, -bounds.min_z).draw(gaussian_stamp)
        _CACHED_GAUSSIAN_ARRAY = Image(gaussian_stamp, Position(bounds.min_x, bounds.min_y, bounds.min_z))

    image_adder.add_images(Image(image, image_position * -1), _CACHED_GAUSSIAN_ARRAY)


def _get_all_positions_without_image_offset(experiment: Experiment, time_point: TimePoint) -> Iterable[Position]:
    """Gets all positions in a time point."""
    offset = experiment.images.offsets.of_time_point(time_point)
    for position in experiment.positions.of_time_point(time_point):
        yield position - offset


def _get_positions_without_image_offset(experiment: Experiment, time_point: TimePoint) -> Iterable[Position]:
    """Yields all positions in a time point."""
    offset = experiment.images.offsets.of_time_point(time_point)

    for position in experiment.positions.of_time_point(time_point):
        yield position - offset


def create_painting_target(config: NetworkConfig, experiment: Experiment, time_point: TimePoint) -> Optional[ndarray]:
    """Draws all positions to the image. Returns None if we cannot determine the correct image size."""
    image_size = _get_image_size(config, experiment, time_point)
    if image_size is None:
        return None

    image = numpy.zeros(image_size)
    division_positions = list(_get_positions_without_image_offset(experiment, time_point))
    for division in division_positions:
        _draw_gaussian(image, division, config.draw_radius_xy, config.draw_radius_z)
    image[image > 1] = 1
    return image


def _get_image_size(config: NetworkConfig, experiment: Experiment, time_point: TimePoint) -> Optional[Tuple[int, int, int]]:
    """Returns the appropriate image size for the particular time point."""
    image_size = experiment.images.image_loader().get_image_size_zyx()

    if image_size is not None:
        return image_size

    # No size known, find it out by loading an image
    transmitted_channel = experiment.images.get_channels()[config.transmitted_light_channel.index_from_zero]
    transmitted_image = experiment.images.get_image_stack(time_point, transmitted_channel)
    if transmitted_image is None:
        return None  # No image known, return nothing
    return transmitted_image.shape
