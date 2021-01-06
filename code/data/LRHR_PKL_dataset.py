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
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch

import pickle


class LRHR_PKLDataset(data.Dataset):
    def __init__(self, opt):
        super(LRHR_PKLDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size", None)
        self.scale = None
        self.random_scale_list = [1]

        hr_file_path = opt["dataroot_GT"]
        lr_file_path = opt["dataroot_LQ"]
        y_labels_file_path = opt['dataroot_y_labels']

        gpu = True
        augment = True

        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)

        n_max = opt["n_max"] if "n_max" in opt.keys() else int(1e8)

        t = time.time()
        self.lr_images = self.load_pkls(lr_file_path, n_max)
        self.hr_images = self.load_pkls(hr_file_path, n_max)

        min_val_hr = np.min([i.min() for i in self.hr_images[:20]])
        max_val_hr = np.max([i.max() for i in self.hr_images[:20]])

        min_val_lr = np.min([i.min() for i in self.lr_images[:20]])
        max_val_lr = np.max([i.max() for i in self.lr_images[:20]])

        t = time.time() - t
        print("Loaded {} HR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.hr_images), min_val_hr, max_val_hr, t, hr_file_path))
        print("Loaded {} LR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.lr_images), min_val_lr, max_val_lr, t, lr_file_path))

        self.gpu = gpu
        self.augment = augment

        self.measures = None

    def load_pkls(self, path, n_max):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        images = images[:n_max]
        images = [np.transpose(image, [2, 0, 1]) for image in images]
        return images

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        hr = self.hr_images[item]
        lr = self.lr_images[item]

        if self.scale == None:
            self.scale = hr.shape[1] // lr.shape[1]
            assert hr.shape[1] == self.scale * lr.shape[1], ('non-fractional ratio', lr.shape, hr.shape)

        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size, self.scale, self.use_crop)

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size // self.scale)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = hr / 255.0
        lr = lr / 255.0

        if self.measures is None or np.random.random() < 0.05:
            if self.measures is None:
                self.measures = {}
            self.measures['hr_means'] = np.mean(hr)
            self.measures['hr_stds'] = np.std(hr)
            self.measures['lr_means'] = np.mean(lr)
            self.measures['lr_stds'] = np.std(lr)

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()

        return {'LQ': lr, 'GT': hr, 'LQ_path': str(item), 'GT_path': str(item)}

    def print_and_reset(self, tag):
        m = self.measures
        kvs = []
        for k in sorted(m.keys()):
            kvs.append("{}={:.2f}".format(k, m[k]))
        print("[KPI] " + tag + ": " + ", ".join(kvs))
        self.measures = None


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg


def random_crop(hr, lr, size_hr, scale, random):
    size_lr = size_hr // scale

    size_lr_x = lr.shape[1]
    size_lr_y = lr.shape[2]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[:, start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]

    # HR Patch
    start_x_hr = start_x_lr * scale
    start_y_hr = start_y_lr * scale
    hr_patch = hr[:, start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr]

    return hr_patch, lr_patch


def center_crop(img, size):
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, border:-border, border:-border]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]
