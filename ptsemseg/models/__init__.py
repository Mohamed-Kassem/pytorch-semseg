import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.linknet import *



def get_model(name, n_classes, kassem=False, exp_index=0):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes, kassem=kassem, exp_index=exp_index)
        #vgg16 = models.vgg16(pretrained=True)
        vgg16 = models.vgg16()
        vgg16.load_state_dict( torch.load('/home/nile002u1/data/models/vgg16-397923af.pth') )
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        #vgg16 = models.vgg16(pretrained=True)
        vgg16 = models.vgg16()
        vgg16.load_state_dict( torch.load('/home/nile002u1/data/models/vgg16-397923af.pth') )
        model.init_vgg16_params(vgg16)

    elif name == 'unet':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)
    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'fcn32s': fcn32s,
        'fcn8s': fcn8s,
        'fcn16s': fcn16s,
        'unet': unet,
        'segnet': segnet,
        'pspnet': pspnet,
        'linknet': linknet,
    }[name]
