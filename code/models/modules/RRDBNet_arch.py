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

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from utils.util import opt_get


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 16:
            self.upconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 32:
            self.upconv5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, get_steps=False):
        fea = self.conv_first(x)

        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_results = {}

        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea

        trunk = self.trunk_conv(fea)

        last_lr_fea = fea + trunk

        fea_up2 = self.upconv1(F.interpolate(last_lr_fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(fea_up2)

        fea_up4 = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(fea_up4)

        fea_up8 = None
        fea_up16 = None
        fea_up32 = None

        if self.scale >= 8:
            fea_up8 = self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(fea_up8)
        if self.scale >= 16:
            fea_up16 = self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(fea_up16)
        if self.scale >= 32:
            fea_up32 = self.upconv5(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(fea_up32)

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        results = {'last_lr_fea': last_lr_fea,
                   'fea_up1': last_lr_fea,
                   'fea_up2': fea_up2,
                   'fea_up4': fea_up4,
                   'fea_up8': fea_up8,
                   'fea_up16': fea_up16,
                   'fea_up32': fea_up32,
                   'out': out}

        fea_up0_en = opt_get(self.opt, ['network_G', 'flow', 'fea_up0']) or False
        if fea_up0_en:
            results['fea_up0'] = F.interpolate(last_lr_fea, scale_factor=1/2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        fea_upn1_en = opt_get(self.opt, ['network_G', 'flow', 'fea_up-1']) or False
        if fea_upn1_en:
            results['fea_up-1'] = F.interpolate(last_lr_fea, scale_factor=1/4, mode='bilinear', align_corners=False, recompute_scale_factor=True)

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return out
