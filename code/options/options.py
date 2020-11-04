import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml

Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    gpu_list = ','.join(str(x) for x in opt.get('gpu_ids', []))
    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        scale = opt['scale']

    # datasets
    for phase, dataset in opt.get('datasets', {}).items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if opt['distortion'] == 'sr':
            dataset['scale'] = scale
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')

    # network
    if opt['distortion'] == 'sr':
        opt['network_G']['scale'] = scale

    # relative learning rate
    if 'train' in opt:
        niter = opt['train']['niter']
        if 'T_period_rel' in opt['train']:
            opt['train']['T_period'] = [int(x * niter) for x in opt['train']['T_period_rel']]
        if 'restarts_rel' in opt['train']:
            opt['train']['restarts'] = [int(x * niter) for x in opt['train']['restarts_rel']]
        if 'lr_steps_rel' in opt['train']:
            opt['train']['lr_steps'] = [int(x * niter) for x in opt['train']['lr_steps_rel']]
        if 'lr_steps_inverse_rel' in opt['train']:
            opt['train']['lr_steps_inverse'] = [int(x * niter) for x in opt['train']['lr_steps_inverse_rel']]
        print(opt['train'])

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
