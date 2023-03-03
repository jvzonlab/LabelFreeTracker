"""This plugin sets the link scores simply based on the distance_um."""
import math
import random
from typing import Dict, Callable, List, Tuple

import numpy
from matplotlib import pyplot as plt
from numpy import ndarray

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io
from organoid_tracker.linking import nearest_neighbor_linker

_EPSILON = 10 ** -10


class _LinkCountsByDistance:

    _distance_bins_um: ndarray
    _count_valid: ndarray
    _count_total: ndarray

    def __init__(self):
        self._distance_bins_um = 10 ** numpy.arange(0, 2, 0.05, dtype=float)
        self._count_valid = numpy.zeros(len(self._distance_bins_um), dtype=numpy.uint32)
        self._count_total = numpy.zeros(len(self._distance_bins_um), dtype=numpy.uint32)

    def add_link(self, distance_um: float, correct: bool):
        bin = 0
        while distance_um > self._distance_bins_um[bin]:
            bin += 1
        self._count_total[bin] += 1
        if correct:
            self._count_valid[bin] += 1

    def ratio(self) -> Tuple[List[float], List[float], List[int]]:
        ratios = list()
        distances = list()
        total_counts = list()
        for i in range(len(self._count_total)):
            count_valid = self._count_valid[i]
            count_total = self._count_total[i]
            if count_total == 0:
                return distances, ratios, total_counts
            ratios.append(float(count_valid) / count_total)
            distances.append(float(self._distance_bins_um[i]))
            total_counts.append(int(count_total))
        return distances, ratios, total_counts

    def get_chance(self, distance_um: float, *, max_chance: float = 0.98) -> float:
        """Gets the chance of a link of the given distance being a real link."""
        bin = 0

        # Iterate until we fall within a bin
        while distance_um > self._distance_bins_um[bin]:
            bin += 1
            if self._count_valid[bin] == 0:
                break  # No use in iterating further

        if bin == 0:  # Special case
            return min(max_chance, self._count_valid[0] / self._count_total[0])

        bin_distance_start = self._distance_bins_um[bin - 1]
        bin_distance_end = self._distance_bins_um[bin]
        bin_fraction = min((distance_um - bin_distance_start) / (bin_distance_end - bin_distance_start), 1)

        fraction_start = float(self._count_valid[bin - 1]) / self._count_total[bin - 1]
        fraction_end = float(self._count_valid[bin]) / self._count_total[bin]

        return min(max_chance, fraction_end * bin_fraction + fraction_start * (1 - bin_fraction))


def get_commands() -> Dict[str, Callable[[List[str]], int]]:
    # Command-line commands, used in python organoid_tracker.py <command>
    return {
        "set_link_scores_from_distance": _link_scores_command,
    }


def _random_offset() -> Position:
    return Position((random.random() - random.random()) * 5, (random.random() - random.random()) * 5,
             random.choice([0, 0, 0, 0, 1, -1]))


def _calculate_link_chance_by_distance(experiment: Experiment) -> _LinkCountsByDistance:
    """Calculates the a, loc and scale of the log-normal distribution fit to all the link distances."""
    link_counts = _LinkCountsByDistance()
    resolution = experiment.images.resolution()
    links_possibilities = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
    for source, target in links_possibilities.find_all_links():
        correct = experiment.links.contains_link(source, target)

        # Add some pertubations
        source = source + _random_offset()
        target = target + _random_offset()

        distance_um = source.distance_um(target, resolution)
        link_counts.add_link(distance_um, correct)

    # To plot: use the following code
    distances, ratios, counts = link_counts.ratio()
    plt.scatter(distances, ratios)
    for distance, ratio, count in zip(distances, ratios, counts):
        plt.text(distance, ratio, str(count))

    distances_small_small_steps = numpy.arange(0, max(distances), 0.01)
    ratios_small_steps = [link_counts.get_chance(distance_um) for distance_um in distances_small_small_steps]
    plt.plot(distances_small_small_steps, ratios_small_steps)

    plt.show()

    return link_counts



def _calculate_division_chance(experiment: Experiment) -> float:
    dividing_positions = 0
    all_positions = 0
    links = experiment.links

    last_time_point_number = experiment.positions.last_time_point_number()
    if last_time_point_number is None:
        return 0
    last_time_point = TimePoint(last_time_point_number)

    for time_point in experiment.positions.time_points():
        if time_point == last_time_point:
            continue  # For this time point, we don't know whether cells are dividing
        for position in experiment.positions.of_time_point(time_point):
            if len(links.find_futures(position)) == 2:
                dividing_positions += 1
            all_positions += 1
    return dividing_positions / all_positions


def _link_scores_command(args: List[str]) -> int:
    # Read config file
    print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
    config = ConfigFile("set_link_scores_from_distance")
    input_file = config.get_or_prompt("input_file", "In which file the positions stored, which will receive link scores?")
    training_file = config.get_or_prompt("training_file", "In which file is training data stored for linking?")
    output_file = config.get_or_default("output_file", "Scored positions.aut", comment="Output file.")
    max_z = config.get_or_default("max_z", "999", comment="Beyond this Z level, positions are deleted. This speeds"
            " up things if you wouldn't need those positions anyways.", type=config_type_int)
    config.save_and_exit_if_changed()

    # Calculate parameters
    print("Fitting distance_um distribution...")
    training_data = io.load_data_file(training_file)
    division_chance = _calculate_division_chance(training_data)
    counts_by_distance = _calculate_link_chance_by_distance(training_data)
    print(f"    found link_chance[d=1-20 um]={[counts_by_distance.get_chance(i + 1) for i in range(20)]}"
          f" and division_chance={division_chance:.3f}")

    # Set link penalties
    print("Calculating link penalties...")
    experiment = io.load_data_file(input_file)
    _delete_high_positions(experiment, max_z)
    _set_division_penalty(experiment, division_chance)
    _set_link_penalty(experiment, counts_by_distance)

    print("Saving file...")
    io.save_data_to_json(experiment, output_file)

    print("Done!")
    return 0


def _delete_high_positions(experiment: Experiment, max_z: int):
    for time_point in experiment.positions.time_points():
        positions = list(experiment.positions.of_time_point(time_point))
        for position in positions:
            if position.z > max_z:
                experiment.remove_position(position)


def _set_division_penalty(experiment: Experiment, division_chance: float):
    penalty = -math.log10(division_chance+_EPSILON)+math.log10(1-division_chance+_EPSILON)

    position_data = experiment.position_data
    for time_point in experiment.positions.time_points():
        for position in experiment.positions.of_time_point(time_point):
            position_data.set_position_data(position, data_name="division_penalty", value=penalty)


def _set_link_penalty(experiment: Experiment, counts_by_distance: _LinkCountsByDistance):
    """Sets a link penalty purely based on the link distance_um."""
    link_data = experiment.link_data

    resolution = experiment.images.resolution()
    experiment.links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
    for source, target in experiment.links.find_all_links():
        chance = counts_by_distance.get_chance(source.distance_um(target, resolution))
        penalty = -math.log10(chance+_EPSILON)+math.log10(1-chance+_EPSILON)
        link_data.set_link_data(source, target, "link_penalty", penalty)
