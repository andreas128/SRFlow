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
from os.path import basename
import math
import argparse
import random
import logging
import cv2

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths


def getEnv(name): import os; return True if name in os.environ.keys() else False


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_deviceDistIterSampler(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = get_resume_paths(opt)

        # distributed resuming: all load into default GPU
        if resume_state_path is None:
            resume_state = None
        else:
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path,
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            conf_name = basename(args.opt).replace(".yml", "")
            exp_dir = opt['path']['experiments_root']
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            print('Dataset created')
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    timer = Timer()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()

    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        timerData.tick()
        for _, train_data in enumerate(train_loader):
            timerData.tock()
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            try:
                nll = model.optimize_parameters(current_step)
            except RuntimeError as e:
                print("Skipping ERROR caught in nll = model.optimize_parameters(current_step): ")
                print(e)

            if nll is None:
                nll = 0

            #### log
            def eta(t_iter):
                return (t_iter * (opt['train']['niter'] - current_step)) / 3600

            if current_step % opt['logger']['print_freq'] == 0 \
                    or current_step - (resume_state['iter'] if resume_state else 0) < 25:
                avg_time = timer.get_average_and_reset()
                avg_data_time = timerData.get_average_and_reset()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, t:{:.2e}, td:{:.2e}, eta:{:.2e}, nll:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate(), avg_time, avg_data_time,
                    eta(avg_time), nll)
                print(message)
            timer.tick()
            # Reduce number of logs
            if current_step % 5 == 0:
                tb_logger_train.add_scalar('loss/nll', nll, current_step)
                tb_logger_train.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                tb_logger_train.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                tb_logger_train.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                tb_logger_train.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)
                for k, v in model.get_current_log().items():
                    tb_logger_train.add_scalar(k, v, current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                nlls = []
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)

                    nll = model.test()
                    if nll is None:
                        nll = 0
                    nlls.append(nll)

                    visuals = model.get_current_visuals()

                    sr_img = None
                    # Save SR images for reference
                    if hasattr(model, 'heats'):
                        for heat in model.heats:
                            for i in range(model.n_sample):
                                sr_img = util.tensor2img(visuals['SR', heat, i])  # uint8
                                save_img_path = os.path.join(img_dir,
                                                             '{:s}_{:09d}_h{:03d}_s{:d}.png'.format(img_name,
                                                                                                    current_step,
                                                                                                    int(heat * 100), i))
                                util.save_img(sr_img, save_img_path)
                    else:
                        sr_img = util.tensor2img(visuals['SR'])  # uint8
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)
                    assert sr_img is not None

                    # Save LQ images for reference
                    save_img_path_lq = os.path.join(img_dir,
                                                    '{:s}_LQ.png'.format(img_name))
                    if not os.path.isfile(save_img_path_lq):
                        lq_img = util.tensor2img(visuals['LQ'])  # uint8
                        util.save_img(
                            cv2.resize(lq_img, dsize=None, fx=opt['scale'], fy=opt['scale'],
                                       interpolation=cv2.INTER_NEAREST),
                            save_img_path_lq)

                    # Save GT images for reference
                    gt_img = util.tensor2img(visuals['GT'])  # uint8
                    save_img_path_gt = os.path.join(img_dir,
                                                    '{:s}_GT.png'.format(img_name))
                    if not os.path.isfile(save_img_path_gt):
                        util.save_img(gt_img, save_img_path_gt)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                avg_psnr = avg_psnr / idx
                avg_nll = sum(nlls) / len(nlls)

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))

                # tensorboard logger
                tb_logger_valid.add_scalar('loss/psnr', avg_psnr, current_step)
                tb_logger_valid.add_scalar('loss/nll', avg_nll, current_step)

                tb_logger_train.flush()
                tb_logger_valid.flush()

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

            timerData.tick()

    with open(os.path.join(opt['path']['root'], "TRAIN_DONE"), 'w') as f:
        f.write("TRAIN_DONE")

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
