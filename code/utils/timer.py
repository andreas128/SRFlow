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

import time


class ScopeTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("{} {:.3E}".format(self.name, self.interval))


class Timer:
    def __init__(self):
        self.times = []

    def tick(self):
        self.times.append(time.time())

    def get_average_and_reset(self):
        if len(self.times) < 2:
            return -1
        avg = (self.times[-1] - self.times[0]) / (len(self.times) - 1)
        self.times = [self.times[-1]]
        return avg

    def get_last_iteration(self):
        if len(self.times) < 2:
            return 0
        return self.times[-1] - self.times[-2]


class TickTock:
    def __init__(self):
        self.time_pairs = []
        self.current_time = None

    def tick(self):
        self.current_time = time.time()

    def tock(self):
        assert self.current_time is not None, self.current_time
        self.time_pairs.append([self.current_time, time.time()])
        self.current_time = None

    def get_average_and_reset(self):
        if len(self.time_pairs) == 0:
            return -1
        deltas = [t2 - t1 for t1, t2 in self.time_pairs]
        avg = sum(deltas) / len(deltas)
        self.time_pairs = []
        return avg

    def get_last_iteration(self):
        if len(self.time_pairs) == 0:
            return -1
        return self.time_pairs[-1][1] - self.time_pairs[-1][0]
