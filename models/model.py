"""
@Time ：2022/7/9 13:32
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
import numpy as np
import torch
import torch.nn as nn
from importlib import import_module
from tools.box_transforms import create_grid


class Yolo(nn.Module):

    def __init__(self, cfg):
        super(Yolo, self).__init__()
        backbone = cfg['backbone']
        self.b = cfg['b']
        self.nc = cfg['nc']
        self.fcn = cfg['fcn']
        x = import_module('models.backbone.' + backbone)
        self.backbone_net = x.get_backbone()
        if self.fcn:
            self.regression = nn.Sequential(
                nn.Conv2d(512, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, 3, 1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, self.b * 5, 3, 1, padding=1),
                nn.BatchNorm2d(self.b * 5),
                nn.Sigmoid(),
            )
            self.classifier = nn.Sequential(
                nn.Conv2d(512, self.nc, 3, 1, padding=1),
                nn.BatchNorm2d(self.nc),
                nn.Sigmoid(),
            )
        else:
            self.regression = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=.2),
                nn.Linear(4096, 7 * 7 * (self.b * 5 + self.nc)),
                nn.Sigmoid()
            )

    def forward(self, inputs):
        feats = self.backbone_net(inputs)
        if self.fcn:
            regs = self.regression(feats)
            cls = self.classifier(feats)
            outs = torch.concat([regs, cls], dim=1)
            outs = torch.permute(outs, (0, 2, 3, 1))
        else:
            outs = self.regression(feats)
            outs = outs.view(-1, 7, 7, self.b * 5 + self.nc)
        return outs

    def decoder(self, batch_outs, threshold=.2, device=None):
        """

        :param outs: （b, 7, 7, 2 * B + num_classes）
        :param threshold:
        :param device:
        :return:
        """
        N, grid_h, grid_w, c = batch_outs.shape

        grid_xy = create_grid(grid_h, grid_w, normalizer=True, device=device)
        batch_boxes = []
        batch_scores = []
        batch_class_ids = []
        for b in range(N):
            outs = batch_outs[b]
            cls_outs = outs[..., 10:]
            box_conf_outs = outs[..., 0:10].view(grid_h, grid_w, 2, 5)

            xy_outs = box_conf_outs[..., 0:2]
            xy_outs += grid_xy[0]
            wh_outs = box_conf_outs[..., 2:4]
            conf_outs = box_conf_outs[..., 4]

            max_conf_outs, index_conf_outs = torch.max(conf_outs, dim=-1)
            boxes = []
            scores = []
            class_ids = []
            for i in range(grid_w):
                for j in range(grid_h):
                    index = index_conf_outs[j, i]
                    max_conf = max_conf_outs[j, i]
                    if max_conf > threshold:
                        prob, cls = torch.max(cls_outs[j, i, :], dim=-1)
                        score = prob * max_conf
                        if score > threshold:
                            class_ids.append(cls.item())
                            scores.append(score.item())
                            if device is not None:
                                xy = xy_outs[j, i, index].cpu().numpy()
                                wh = wh_outs[j, i, index].cpu().numpy()
                            else:
                                xy = xy_outs[j, i, index].numpy()
                                wh = wh_outs[j, i, index].numpy()
                            boxes.append(np.concatenate([xy, wh], axis=0))

            boxes = np.array(boxes)
            scores = np.array(scores)
            class_ids = np.array(class_ids)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_class_ids.append(class_ids)

        batch_boxes = np.array(batch_boxes)
        batch_scores = np.array(batch_scores)
        batch_class_ids = np.array(batch_class_ids)
        return batch_boxes, batch_scores, batch_class_ids
