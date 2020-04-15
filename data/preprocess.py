import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import roberts


def preprocess(image_dir, mask_dir, channel=1, width=288):
    path = list(zip(_get_path(image_dir), _get_path(mask_dir)))
    image = np.zeros(shape=(len(path), channel, width, width))
    mask = np.zeros(shape=(len(path), channel, width, width))
    for i, p in tqdm(enumerate(path), total=len(path), desc='Load data', unit=' image/mask(s)'):
        image[i], mask[i] = _roi(*p)
    return image, mask


def roi(image_path, mask_path):
    return _roi(image_path, mask_path)


def _roi(image_path, mask_path):
    image = plt.imread(image_path)  # 1 channel
    mask = plt.imread(mask_path)  # 3 channels
    mask = mask[:, :, 0]  # 3 channels --> 1 channel
    mask[np.where(mask > 0)] = 1  # 0.502 --> 1

    x, y = _get_centre(image)
    w = 144  # width 288
    roi = [image[y - w:y + w, x - w:x + w]]  # 288×288 --> 1×288×288
    gt = [mask[y - w:y + w, x - w:x + w]]  # ground truth

    if x + w > image.shape[1] - 1:  # exceptional samples
        roi = [image[y - w:y + w, -w * 2:]]
        gt = [mask[y - w:y + w, -w * 2:]]

    # fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    # ax = axes.ravel()
    # ax[0].imshow(roi[0], cmap=plt.cm.gray)
    # ax[1].imshow(gt[0], cmap='binary')

    # plt.show()

    return roi, gt


def _get_path(dir):
    dir = os.path.realpath(dir)
    files = os.listdir(dir)
    path = []
    for file in files:
        path.append(os.path.join(dir, file))
    return path


def _get_centre(image):
    edge = roberts(image)
    centre = np.where(edge > 0.45)
    x = centre[1][0]
    y = centre[0][0]
    return x, y
