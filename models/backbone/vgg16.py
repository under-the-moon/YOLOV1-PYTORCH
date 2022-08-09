"""
@Time ：2022/7/9 13:32
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
from models.backbone.vgg import vgg16, VGG16_Weights


def get_backbone():
    return vgg16(VGG16_Weights)
