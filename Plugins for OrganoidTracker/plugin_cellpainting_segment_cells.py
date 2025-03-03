import os
from typing import Any, Iterable, Tuple, List, Dict

import numpy
import scipy
import skimage
import tifffile
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob, submit_job


def get_menu_items(window: Window) -> Dict[str, Any]:
    # Menu options for the OrganoidTracker GUI
    return {
        "Tools//Cellpainting-From transmitted light//Segmentation-Segment cells...": lambda: _segment_cells(window)
    }


def _segment_cells(window: Window):
    experiment_names = set()
    for experiment in window.get_active_experiments():
        if experiment.name.get_save_name() in experiment_names:
            raise UserError("Duplicate experiment name", f"The experiment name \"{experiment.name}\" was"
                            f" found multiple times. Please give each experiment a unique name.")
        if not experiment.positions.has_positions():
            raise UserError("No positions found",
                            f"No nucleus center positions found in experiment {experiment.name}. Please run"
                            f" the nucleus center position prediction first, and then load the data.")
        experiment.images.resolution()  # Checks whether a resolution is set
    if not dialog.popup_message_cancellable("Cell segmentation", "This will segment cells using a watershed"
                             " transformation on the predicted membrane signal. Make sure you have the predicted"
                             " membrane channel set as the current channel."):
        return

    membrane_channel = window.display_settings.image_channel
    output_folder = dialog.prompt_save_file("Select an output folder", [("Folder", "*")])
    if output_folder is None:
        return  # Cancelled
    if os.path.exists(output_folder) and not os.path.isdir(output_folder):
        raise UserError("Not an output folder", f"Please choose an output folder. \"{output_folder}\" is a file.")

    submit_job(window, _SegmentJob(output_folder, len(experiment_names) > 0, membrane_channel))


class _SegmentJob(WorkerJob):

    _output_folder: str
    _multiple_experiments: bool
    _painted_membrane_channel: ImageChannel

    def __init__(self, output_folder: str, multiple_experiments: bool, painted_membrane_channel: ImageChannel):
        self._output_folder = output_folder
        self._multiple_experiments = multiple_experiments
        self._painted_membrane_channel = painted_membrane_channel

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, positions=True, name=True)

    def _get_output_folder(self, experiment: Experiment) -> str:
        if not self._multiple_experiments:
            return self._output_folder
        return os.path.join(self._output_folder, experiment.name.get_save_name())

    def gather_data(self, experiment_copy: Experiment) -> Any:
        experiment = experiment_copy
        output_folder = self._get_output_folder(experiment)
        os.makedirs(output_folder, exist_ok=True)

        for time_point in experiment.positions.time_points():
            output_file = os.path.join(output_folder, f"watershed_{time_point.time_point_number()}.tif")
            if os.path.exists(output_file):
                print("    skipping, as file already exists")
                continue

            membrane_image = experiment.images.get_image_stack(time_point, self._painted_membrane_channel)
            positions = list(experiment.positions.of_time_point(time_point))
            resolution = experiment.images.resolution()

            membrane_image, scale_factor = _scale_image(membrane_image, experiment.images.resolution())
            scaled_positions = [Position(p.x, p.y, p.z * scale_factor, time_point=time_point) for p in positions]
            resolution = ImageResolution(resolution.pixel_size_x_um, resolution.pixel_size_y_um,
                                         resolution.pixel_size_z_um / scale_factor, resolution.time_point_interval_m)
            image_offset = experiment.images.offsets.of_time_point(time_point)
            scaled_image_offset = Position(image_offset.x, image_offset.y, image_offset.z * scale_factor, time_point=time_point)

            watershed_image = _create_watershed_image(membrane_image, scaled_positions, scaled_image_offset, resolution)

            # Make the image indexed by track_id
            watershed_image_correct_colors = _index_by_track_id(experiment, time_point, positions, watershed_image)
            del watershed_image

            tifffile.imwrite(output_file, watershed_image_correct_colors,
                             compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

    def on_finished(self, data: Iterable[Any]):
        if dialog.prompt_options("Segmentation complete", "Done! Segmented all time points.",
                                 option_1="Open output folder", option_default=DefaultOption.OK):
            dialog.open_file(self._output_folder)


def _is_out_of_bounds(coords: Tuple[int, ...], image: ndarray) -> bool:
    """Checks if the given coord is out of bounds for the given array."""
    for i in range(len(coords)):
        if coords[i] < 0 or coords[i] >= image.shape[i]:
            return True
    return False


def _get_organoid_mask(membrane_image: ndarray) -> ndarray:
    # Threshold the membrane, fill the holes
    membrane_mask_image: ndarray = membrane_image > membrane_image.max() / 10
    for z in range(membrane_mask_image.shape[0]):
        membrane_mask_image[z] = scipy.ndimage.binary_dilation(membrane_mask_image[z], iterations=10)
        membrane_mask_image[z] = scipy.ndimage.binary_erosion(membrane_mask_image[z], iterations=10)

    # Remove everything but the largest object
    membrane_mask_image = skimage.measure.label(membrane_mask_image)
    sizes = numpy.bincount(membrane_mask_image.flatten())
    largest_object = int(numpy.argmax(sizes[1:]) + 1)
    membrane_mask_image = (membrane_mask_image == largest_object).astype(numpy.uint8)

    # Make the mask a bit smaller
    for z in range(membrane_mask_image.shape[0]):
        membrane_mask_image[z] = scipy.ndimage.binary_erosion(membrane_mask_image[z], structure=numpy.ones((8, 8)))

    return membrane_mask_image


def _create_watershed_image(membrane_image: ndarray, positions: List[Position], image_offset: Position, resolution: ImageResolution) -> ndarray:
    membrane_image = membrane_image.astype(numpy.float32)  # We don't want to modify the original array in this function
    membrane_image /= membrane_image.max()

    # Create positions image, from which we start the watershed
    positions_image = numpy.zeros(membrane_image.shape, dtype=numpy.uint16)  # Small numbered squares at positions
    hole_at_positions_image = numpy.ones_like(membrane_image, dtype=numpy.uint8)  # 1 everywhere, except at positions
    for i, position in enumerate(positions):
        x = int(position.x - image_offset.x)
        y = int(position.y - image_offset.y)
        z = int(position.z - image_offset.z)
        if _is_out_of_bounds((z, y, x), positions_image):
            continue  # Outside image
        positions_image[z - 4: z+4, y - 4:y + 4, x - 4:x + 4] = i + 1
        hole_at_positions_image[z, y, x] = 0

    # Do the distance transform (from 0 to 1)
    distance_landscape = scipy.ndimage.distance_transform_edt(input=hole_at_positions_image,
                                                              sampling=resolution.pixel_size_zyx_um)
    max_radius_um = 7
    numpy.clip(distance_landscape, a_min=0, a_max=max_radius_um, out=distance_landscape)
    distance_landscape /= (max_radius_um * 3)  # So goes from 0 to 0.33

    # Do not watershed above the distance threshold
    outside_organoid_value = positions_image.max() + 1
    positions_image[_get_organoid_mask(membrane_image) == 0] = outside_organoid_value

    # Watershed!
    watershed_image: ndarray = skimage.segmentation.watershed(distance_landscape + membrane_image, markers=positions_image)
    watershed_image[watershed_image == outside_organoid_value] = 0
    return watershed_image.astype(numpy.int16)


def _scale_image(multi_im: ndarray, resolution: ImageResolution) -> Tuple[ndarray, float]:
    """Attempts to scale the image such that the z-resolution is equal to the xy-resolution. Assumes that the x and
    y resolutions are already equal, and that the z-resolution is lower (higher number) than the xy-resolution.

    Returns the rescaled image, and the scale factor (which is a whole number). Note: we only add extra layers in
    between, we don't modify layers from the input image. Therefore, the resolution xy and z scale might still be
    slightly different, so the returned scale factor isn't simply resolution.z / resolution.x.
    """

    mid_layers_nb = int((resolution.pixel_size_z_um / resolution.pixel_size_x_um) - 1)

    if mid_layers_nb < 0:
        raise ValueError("negative number of mid layers, z-res was higher than x-res?")
    if mid_layers_nb == 0:
        return multi_im, 1  # No need to reconstruct anything

    out_img = numpy.zeros(
        (int(len(multi_im) + mid_layers_nb * (len(multi_im) - 1) + 2 * mid_layers_nb),) + multi_im[0].shape).astype(
        multi_im[0].dtype)

    layer_index = mid_layers_nb + 1
    orig_index = []

    for i in range(len(multi_im) - 1):

        for layer in range(mid_layers_nb + 1):
            t = float(layer) / (mid_layers_nb + 1)
            interpolate = ((1 - t) * (multi_im[i]).astype(float) + t * (multi_im[i + 1]).astype(float))

            out_img[layer_index] = interpolate

            if t == 0:
                orig_index.append(layer_index)
            layer_index += 1

    return out_img, mid_layers_nb + 1


def _index_by_track_id(experiment: Experiment, time_point: TimePoint, positions: List[Position], watershed_image: ndarray) -> ndarray:
    if experiment.links.has_links():
        watershed_image_correct_colors = numpy.zeros_like(watershed_image, dtype=numpy.uint16)

        # Outline organoid mask
        watershed_image_correct_colors[7:][watershed_image[7:] > 0] = 1

        # Add back ids
        for id, track in experiment.links.find_all_tracks_and_ids():
            if time_point.time_point_number() >= track.first_time_point_number() and time_point.time_point_number() <= track.last_time_point_number():
                position = track.find_position_at_time_point_number(time_point.time_point_number())
                try:
                    old_index = positions.index(position) + 1
                    watershed_image_correct_colors[watershed_image == old_index] = id + 2
                except IndexError:
                    pass  # Position is not in experiment.positions for whatever reason

        return watershed_image_correct_colors
    else:
        return watershed_image.astype(numpy.int16)
