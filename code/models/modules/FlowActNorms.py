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
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
from torch import nn as nn

from models.modules import thops


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(logs) * thops.pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale

            # self.input = input
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class MaskedActNorm2d(ActNorm2d):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def forward(self, input, mask, logdet=None, reverse=False):

        assert mask.dtype == torch.bool
        output, logdet_out = super().forward(input, logdet, reverse)

        input[mask] = output[mask]
        logdet[mask] = logdet_out[mask]

        return input, logdet

