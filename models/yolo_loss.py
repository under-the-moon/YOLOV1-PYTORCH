import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.box_transforms import calc_iou, convert_to_corners_tensor, create_grid


class YoloLoss(nn.Module):
    def __init__(self, grid, b, gama_coord, gama_noobj, device):
        super(YoloLoss, self).__init__()
        self.grid = grid
        self.b = b
        self.gama_coord = gama_coord
        self.gama_noobj = gama_noobj
        self.device = device

    def forward(self, preds, targets):
        """

        :param preds: (b, grid, grid, b * 5 + num_classes)
        :param targets: (b, grid, grid, 5 + num_classes)
        :return:
        """
        N, grid_h, grid_w, c = preds.shape
        # b, 7, 7, 1
        obj_mask = targets[..., 4:5]
        conf_targets = obj_mask
        conf_targets = conf_targets.unsqueeze(dim=-2)
        box_targets = targets[..., 0:4]
        box_targets = box_targets.unsqueeze(dim=3).tile((1, 1, 1, 2, 1))
        cls_targets = targets[..., 5:]

        conf_box_preds = preds[..., 0:10].view(N, grid_h, grid_w, self.b, 5)
        conf_preds = conf_box_preds[..., 4:5]
        box_preds = conf_box_preds[..., 0:4]
        cls_preds = preds[..., 10:]

        # calc cls loss
        cls_targets = obj_mask * cls_targets
        cls_preds = obj_mask * cls_preds
        cls_loss = F.mse_loss(cls_preds, cls_targets, size_average=False)

        # calc response obj loss
        grid_xy = create_grid(grid_h, grid_w, normalizer=True, device=self.device)
        preds_xy = box_preds[..., 0:2]
        targets_xy = box_targets[..., 0:2]

        preds_xy += grid_xy
        targets_xy += grid_xy

        preds_xywh = torch.concat([preds_xy, box_preds[..., 2:4]], dim=-1)
        targets_xywh = torch.concat([targets_xy, box_targets[..., 2:4]], dim=-1)

        preds_xyxy = convert_to_corners_tensor(preds_xywh)
        targets_xyxy = convert_to_corners_tensor(targets_xywh)

        iou = calc_iou(preds_xyxy, targets_xyxy)

        # (b, 7, 7, 1)
        response_mask = torch.argmax(iou, dim=-1, keepdim=True) * obj_mask
        response_mask = response_mask.unsqueeze(dim=-2)
        no_response_mask = 1 - response_mask

        xy_loss = F.mse_loss(response_mask * box_preds[..., 0:2], response_mask * box_targets[..., 0:2],
                             size_average=False)
        wh_loss = F.mse_loss(response_mask * torch.sqrt(box_preds[..., 2:4]),
                             response_mask * torch.sqrt(box_targets[..., 2:4]), size_average=False)

        conf_loss = F.mse_loss(response_mask * conf_preds, response_mask * conf_targets, size_average=False) + \
                    self.gama_noobj * F.mse_loss(no_response_mask * conf_preds, no_response_mask * conf_targets,
                                                 size_average=False)
        loss = cls_loss + self.gama_coord * xy_loss + self.gama_coord * wh_loss + conf_loss
        return loss / N
