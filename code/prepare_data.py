# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import sys

import numpy as np
import random
import imageio
import pickle

from natsort import natsort
from tqdm import tqdm

def get_img_paths(dir_path, wildcard='*.png'):
    return natsort.natsorted(glob.glob(dir_path + '/' + wildcard))

def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)

def to_pklv4(obj, path, vebose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if vebose:
        print("Wrote {}".format(path))


from imresize import imresize

def random_crop(img, size):
    h, w, c = img.shape

    h_start = np.random.randint(0, h - size)
    h_end = h_start + size

    w_start = np.random.randint(0, w - size)
    w_end = w_start + size

    return img[h_start:h_end, w_start:w_end]


def imread(img_path):
    img = imageio.imread(img_path)
    if len(img.shape) == 2:
        img = np.stack([img, ] * 3, axis=2)
    return img


def to_pklv4_1pct(obj, path, vebose):
    n = int(round(len(obj) * 0.01))
    path = path.replace(".", "_1pct.")
    to_pklv4(obj[:n], path, vebose=True)


def main(dir_path):
    hrs = []
    lqs = []

    img_paths = get_img_paths(dir_path)
    for img_path in tqdm(img_paths):
        img = imread(img_path)

        for i in range(47):
            crop = random_crop(img, 160)
            cropX4 = imresize(crop, scalar_scale=0.25)
            hrs.append(crop)
            lqs.append(cropX4)

    shuffle_combined(hrs, lqs)

    hrs_path = get_hrs_path(dir_path)
    to_pklv4(hrs, hrs_path, vebose=True)
    to_pklv4_1pct(hrs, hrs_path, vebose=True)

    lqs_path = get_lqs_path(dir_path)
    to_pklv4(lqs, lqs_path, vebose=True)
    to_pklv4_1pct(lqs, lqs_path, vebose=True)


def get_hrs_path(dir_path):
    base_dir = os.path.dirname(dir_path)
    name = os.path.basename(dir_path)
    hrs_path = os.path.join(base_dir, 'pkls', name + '.pklv4')
    return hrs_path


def get_lqs_path(dir_path):
    base_dir = os.path.dirname(dir_path)
    name = os.path.basename(dir_path)
    hrs_path = os.path.join(base_dir, 'pkls', name + '_X4.pklv4')
    return hrs_path


def shuffle_combined(hrs, lqs):
    combined = list(zip(hrs, lqs))
    random.shuffle(combined)
    hrs[:], lqs[:] = zip(*combined)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    assert os.path.isdir(dir_path)
    main(dir_path)
