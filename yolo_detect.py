import cv2
import os
import numpy as np
import torch
import colorsys
from tools.visualization import save_img, show_img
from tools.box_transforms import convert_to_corners
from tools.nms import nms


class YoloDetect:

    def __init__(self, model, names, img_size=224, threshold=.4, device=None):
        self.model = model
        self.names = names
        self.img_size = img_size
        self.threshold = threshold
        self.device = device
        self.colors = self.init_colors(len(names))

        self.model.eval()
        if self.device is not None:
            self.model.to(self.device)

    def init_colors(self, n):
        hsv_tuples = [(x / n, 1., 1.)
                      for x in range(n)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)
        return colors

    def _parse_img_paths(self, imgs_path):
        imgs = []
        pre_imgs = []
        imgs_name = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            pre_imgs.append(img)
            h, w = img.shape[0:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
            imgs_name.append(os.path.basename(img_path))
        imgs = np.array(imgs, dtype=np.float32)
        imgs = torch.from_numpy(imgs)
        if self.device is not None:
            imgs = imgs.to(self.device)
        return imgs, pre_imgs, imgs_name

    def detect_imgs(self, img_paths, save=False):
        save_dir = 'run/detect'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n = len(img_paths)
        imgs, pre_imgs, imgs_name = self._parse_img_paths(img_paths)
        with torch.no_grad():
            outs = self.model(imgs)
        batch_boxes, batch_scores, batch_class_ids = self.model.decoder(outs, self.threshold, self.device)
        for i in range(n):
            pre_img = pre_imgs[i]
            h, w = pre_img.shape[0:2]
            boxes = batch_boxes[i]
            num = boxes.shape[0]
            if num == 0:
                if save:
                    save_path = os.path.join(save_dir, f'{imgs_name[i]}')
                    cv2.imwrite(save_path, pre_imgs[i])
                else:
                    cv2.imshow('image', pre_imgs[i])
                    cv2.waitKey(0)
                continue
            boxes = boxes * np.array([w, h, w, h])
            boxes = convert_to_corners(boxes)
            scores = batch_scores[i]
            class_ids = batch_class_ids[i]
            # nms
            keep = nms(boxes, scores)
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
            if save:
                # image, boxes, scores, class_ids, names, colors

                save_path = os.path.join(save_dir, f'{imgs_name[i]}')
                save_img(pre_img, boxes, scores, class_ids, self.names, self.colors, save_path)
            else:
                show_img(pre_img, boxes, scores, class_ids, self.names, self.colors)
            # 185,62,279,199,14 90,78,403,336,12

    def detect_img(self, img_path, save=False):
        img_paths = [img_path]
        self.detect_imgs(img_paths, save=save)


if __name__ == '__main__':
    from models.model import Yolo

    names = ['aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor']
    model = Yolo({'backbone': 'vgg16', 'fcn': True, 'b': 2, 'nc': 20})

    model.load_state_dict(torch.load('run/logs/weights/yolo_best.pth'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    detect = YoloDetect(model, names=names, device=device)
    detect.detect_img('D:/competition_data/OBJECT_DETECTION/PascalVoc/VOCdevkit/VOC2007/JPEGImages/000077.jpg',
                      save=True)
