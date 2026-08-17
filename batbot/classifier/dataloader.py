"""Image loading and windowing for the BatBot species classifier."""

import os

import cv2
import numpy as np

BATCH_SIZE = int(os.getenv('CLASSIFIER_BATCH_SIZE', 10))
INPUT_SIZE = 224
WINDOW_STRIDE = 100
HORIZONTAL_SCALE = 0.5


def _load_image(filepath):
    """Load a spectrogram in the BGR byte layout used to train the model."""
    image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError('Unable to load spectrogram: {}'.format(filepath))
    return image


def _prepare_image(
    image,
    input_size=INPUT_SIZE,
    window_stride=WINDOW_STRIDE,
    horizontal_scale=HORIZONTAL_SCALE,
):
    """Resize a spectrogram and split it into overlapping square windows.

    The training-time evaluation script resized a 300-pixel-high image by
    ``224 / 300`` vertically and half that amount horizontally.  Computing the
    vertical scale from the actual image height retains that transform for the
    original data while also accepting the spectrogram height emitted by
    BatBot today.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a three-channel spectrogram image')
    if input_size <= 0:
        raise ValueError('input_size must be positive')
    if window_stride <= 0:
        raise ValueError('window_stride must be positive')

    height, width, _ = image.shape
    ratio_y = input_size / float(height)
    target_width = max(1, int(round(width * ratio_y * horizontal_scale)))
    resized = cv2.resize(
        image,
        (target_width, input_size),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # The reference inference script pads narrow inputs to one pixel wider than
    # a square so that range(0, width - height, stride) yields one window.
    if resized.shape[1] <= input_size:
        canvas = np.zeros((input_size, input_size + 1, 3), dtype=resized.dtype)
        canvas[:, : resized.shape[1], :] = resized
        resized = canvas

    starts = range(0, resized.shape[1] - input_size, window_stride)
    windows = [resized[:, start : start + input_size, :] for start in starts]
    if not windows:  # Defensive fallback for custom transform arguments.
        windows = [resized[:, :input_size, :]]

    return np.ascontiguousarray(np.stack(windows), dtype=np.uint8)


def _init_transforms(**kwargs):
    """Return the deterministic preprocessing transform used for inference."""

    def transform(image):
        return _prepare_image(image, **kwargs)

    return transform


class ImageFilePathList:
    """Small, dependency-free equivalent of Scoutbot's image path dataset."""

    def __init__(self, filepaths, targets=None, transform=None, target_transform=None):
        self.filepaths = [str(filepath) for filepath in filepaths]
        self.target_values = list(targets) if targets is not None else None
        if self.target_values is not None and len(self.filepaths) != len(self.target_values):
            raise ValueError('filepaths and targets must have the same length')

        self.loader = _load_image
        self.transform = transform
        self.target_transform = target_transform

        if self.target_values is None:
            self.classes, self.class_to_idx = None, None
        else:
            self.classes = sorted(set(self.target_values))
            self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}

    def __getitem__(self, index):
        sample = self.loader(self.filepaths[index])
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_values is None:
            return (sample,)

        target = self.target_values[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.filepaths)
