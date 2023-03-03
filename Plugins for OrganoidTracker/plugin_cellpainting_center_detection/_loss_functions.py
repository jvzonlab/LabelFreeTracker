from typing import Tuple, List

import tensorflow


def _disk(range_zyx: Tuple[float, float, float] = (2.5, 11., 11.), n_channels: int = 1):
    range_int = tensorflow.floor(range_zyx)

    z = tensorflow.range(-range_int[0], range_int[0] + 1) / range_zyx[0]
    y = tensorflow.range(-range_int[1], range_int[1] + 1) / range_zyx[1]
    x = tensorflow.range(-range_int[2], range_int[2] + 1) / range_zyx[2]

    Z, Y, X = tensorflow.meshgrid(z, x, y, indexing='ij')

    distance = tensorflow.square(Z) + tensorflow.square(Y) + tensorflow.square(X)

    disk = tensorflow.where(distance < 1, 1., 0.)

    disk = tensorflow.expand_dims(disk, axis=-1)
    return tensorflow.expand_dims(tensorflow.tile(disk, (1, 1, 1, n_channels)), axis=-1)


def _disk_labels(label, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.)):
    """Changes the labels (single pixels) into disks."""
    disk = _disk(range_zyx= range_zyx)

    label_disk = tensorflow.nn.conv3d(label, disk, [1, 1, 1, 1, 1], 'SAME')

    return label_disk


def peak_finding(y_pred: tensorflow.Tensor, threshold: float = 0.1, volume: List[int] =[3, 13, 13]):
    """Finds the local peaks in the given predictions. Operates by doing a dilation,
    and then checking where the actual value reaches the dilation."""
    dilation = tensorflow.nn.max_pool3d(y_pred, ksize=volume, strides=1, padding='SAME')
    peaks = tensorflow.where(dilation == y_pred, 1., 0)
    range = tensorflow.reduce_max(y_pred) - tensorflow.reduce_min(y_pred)
    peaks = tensorflow.where(y_pred > threshold*range, peaks, 0)
    return peaks


def position_precision(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor):
    """"""
    peaks = peak_finding(y_pred)

    edges = get_edges(peaks)
    peaks = tensorflow.where(edges == 0, peaks, 0)

    y_true_blur = _disk_labels(y_true)
    correct_peaks = tensorflow.where(y_true_blur > 0, peaks, 0)

    return (tensorflow.reduce_sum(correct_peaks)+0.01)/(tensorflow.reduce_sum(peaks)+0.01)


def get_edges(peaks):
    edges = tensorflow.zeros(tensorflow.shape(peaks))
    edges = tensorflow.pad(edges, paddings=[[0, 0], [0, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
    edges = _disk_labels(edges)
    edges = edges[:, 0:-1, 1:-1, 1:-1, :]

    return edges


def position_recall(y_true, y_pred):
    peaks = peak_finding(y_pred)

    positions = peak_finding(y_true)

    peaks_blur = _disk_labels(peaks)
    detected_positions = tensorflow.where(peaks_blur > 0, positions, 0)

    return tensorflow.reduce_sum(detected_positions) / tensorflow.reduce_sum(positions)


def overcount(y_true, y_pred):
    peaks = peak_finding(y_pred)

    correct_positions = tensorflow.where(y_true > 0, peaks, 0)

    positions = peak_finding(y_true, volume=[1, 3, 3], threshold=0.1)

    return tensorflow.reduce_sum(correct_positions) / tensorflow.reduce_sum(positions)


@tensorflow.function
def weighted_loss(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    """Calculates the loss for two images, weighted for the amount of nonzero pixels."""

    # weight the loss by the amount of non zeroes values in label
    weights = tensorflow.where(tensorflow.equal(y_true, 0), 0.25, 1)

    # Calculate weighted mean square error
    squared_difference = tensorflow.square(y_true - y_pred)
    squared_difference = tensorflow.multiply(weights, squared_difference)
    return tensorflow.reduce_mean(squared_difference, axis=[1, 2, 3, 4])
