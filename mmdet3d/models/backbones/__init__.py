# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt

from .._import_utils import optional_import

__all__ = ['ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet']

optional_import('.cylinder3d', ['Asymm3DSpconv'], globals(), __all__)
optional_import('.dgcnn', ['DGCNNBackbone'], globals(), __all__)
optional_import('.dla', ['DLANet'], globals(), __all__)
optional_import('.mink_resnet', ['MinkResNet'], globals(), __all__)
optional_import('.minkunet_backbone', ['MinkUNetBackbone'], globals(), __all__)
optional_import('.multi_backbone', ['MultiBackbone'], globals(), __all__)
optional_import('.nostem_regnet', ['NoStemRegNet'], globals(), __all__)
optional_import('.pointnet2_sa_msg', ['PointNet2SAMSG'], globals(), __all__)
optional_import('.pointnet2_sa_ssg', ['PointNet2SASSG'], globals(), __all__)
optional_import('.second', ['SECOND'], globals(), __all__)
optional_import('.spvcnn_backone',
                ['MinkUNetBackboneV2', 'SPVCNNBackbone'], globals(), __all__)
