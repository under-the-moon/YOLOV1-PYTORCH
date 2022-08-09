"""
@Time ：2022/7/9 16:51
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
import numpy as np
import torch


def convert_to_xywh(boxes):
    """
    x1,y1,x2,y2 -> x_center,ycenter,w,h
    :param boxes:
    :return:
    """

    return np.concatenate(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """
    xywh -> x1y1x2y2
    :param boxes:
    :return:
    """
    return np.concatenate(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,

    )


def convert_to_xywh_tensor(boxes):
    """
    x1,y1,x2,y2 -> x_center,ycenter,w,h
    :param boxes:
    :return:
    """

    return torch.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        dim=-1,
    )


def convert_to_corners_tensor(boxes):
    """
    xywh -> x1y1x2y2
    :param boxes:
    :return:
    """
    return torch.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        dim=-1,

    )


def calc_iou(pred_boxes, target_boxes):
    """

    :param pred_boxes: (..., 2, 4)
    :param target_boxes: (..., 1, 4)
    :return:
    """
    lu = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    rd = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])

    intersection = torch.maximum(torch.zeros_like(rd - lu), rd - lu)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    pred_boxes_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    target_boxes_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    union_area = torch.maximum(
        pred_boxes_area + target_boxes_area - intersection_area, torch.tensor(1e-8)
    )
    return torch.clip(intersection_area / union_area, 0.0, 1.0)


def create_grid(grid_h, grid_w, normalizer=False, device=None):
    grid_y, grid_x = torch.meshgrid([torch.arange(0, grid_h), torch.arange(0, grid_w)])
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    grid_xy = grid_xy.unsqueeze(dim=0).unsqueeze(dim=-2)
    if normalizer:
        grid_xy[..., 0] /= grid_w
        grid_xy[..., 1] /= grid_h
    if device is not None:
        grid_xy = grid_xy.to(device)
    return grid_xy
