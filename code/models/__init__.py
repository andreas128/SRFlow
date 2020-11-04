import importlib
import logging
import os

try:
    import local_config
except:
    local_config = None


logger = logging.getLogger('base')


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'Model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s." % (
            model_filename, target_model_name))
        exit(0)

    return model


def create_model(opt, step=0, **opt_kwargs):
    if local_config is not None:
        opt['path']['pretrain_model_G'] = os.path.join(local_config.checkpoint_path, os.path.basename(opt['path']['results_root'] + '.pth'))

    for k, v in opt_kwargs.items():
        opt[k] = v

    model = opt['model']

    M = find_model_using_name(model)

    m = M(opt, step)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
