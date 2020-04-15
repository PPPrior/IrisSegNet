from backbone import vgg
from ._utils import IntermediateLayerGetter
from .fcn import FCN, FCNHead, SkipArch

__all__ = ['fcn_vgg16', 'fcn_vgg19']


def _segm_resnet(backbone_name, num_classes, arch, pretrained_backbone=False):
    backbone = vgg.__dict__[backbone_name](
        pretrained=pretrained_backbone)

    return_layers = {'pool5': 'out'}
    if arch:
        return_layers['pool4'] = 'pool4'
        return_layers['pool3'] = 'pool3'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 512
    classifier = FCNHead(inplanes, num_classes)

    skip_arch_0 = None  # layer3: inplanes = 512
    skip_arch_1 = None  # layer2: inplanes = 256
    if arch:
        aux_classifier_0 = FCNHead(inplanes, num_classes)
        skip_arch_0 = SkipArch(aux_classifier_0, num_classes)
        if arch == 'fcn8s':
            inplanes = 256
            aux_classifier_1 = FCNHead(inplanes, num_classes)
            skip_arch_1 = SkipArch(aux_classifier_1, num_classes)

    model = FCN(backbone, classifier, skip_arch_0, skip_arch_1)
    return model


def fcn_vgg16(num_classes=1, arch_type=None, **kwargs):
    '''
        Constructs a Fully-Convolutional Network model with a Vgg19 backbone.
    '''
    return _segm_resnet('vgg16', num_classes, arch_type, **kwargs)


def fcn_vgg19(num_classes=1, arch_type=None, **kwargs):
    '''
        Constructs a Fully-Convolutional Network model with a Vgg19 backbone.
    '''
    return _segm_resnet('vgg19', num_classes, arch_type, **kwargs)
