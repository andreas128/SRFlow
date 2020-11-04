import importlib
import logging

logger = logging.getLogger('base')


def find_model_using_name(model_name):
    model_filename = "models.modules." + model_name + "_arch"
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace('_Net', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s." % (
                model_filename, target_model_name))
        exit(0)

    return model

def define_Flow(opt, step):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    Arch = find_model_using_name(which_model)
    netG = Arch(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt['scale'], K=opt_net['flow']['K'], opt=opt, step=step)
    return netG
