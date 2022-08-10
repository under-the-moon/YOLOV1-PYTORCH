import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from tools.box_transforms import convert_to_xywh


class MyDataset(Dataset):

    def __init__(self, lines, img_size=224, grid=7, b=2, num_classes=20, transform=None, mean=None, shuffle=False):
        self.lines = lines
        self.img_size = img_size
        self.grid = grid
        self.b = b
        self.num_classes = num_classes
        self.transform = transform
        self.mean = mean
        if shuffle:
            np.random.shuffle(self.lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        line_arr = line.split(" ")
        image_path = line_arr[0]
        boxes = np.array([list(map(float, b.split(','))) for b in line_arr[1:]])

        img = cv2.imread(image_path)
        h, w = img.shape[0:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            data = {'img': img, 'boxes': boxes}
            t_data = self.transform(data)
            img = t_data['img']
            h, w = img.shape[0:2]
            boxes = t_data['boxes']
        # boxes = boxes
        xywh = convert_to_xywh(boxes[:, 0:4])
        xywh[:, 0:4] /= np.array([w, h, w, h], dtype=np.float32)
        xywhc = np.concatenate([xywh, boxes[:, 4:]], axis=-1)
        target = self.get_target(xywhc, self.grid)
        if self.mean is not None:
            img = img - np.array(self.mean)
        img = img / 255
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(target)

    def get_target(self, boxes, grid):
        target = np.zeros((grid, grid, self.b * 5 + self.num_classes))
        for box in boxes:
            x, y, w, h, c = box
            c = int(c)
            i = int(np.floor(x * grid))
            j = int(np.floor(y * grid))
            delta_x = x - i / grid
            delta_y = y - j / grid
            target[j, i, c + self.b * 5] = 1
            xywh = [delta_x, delta_y, w, h]
            target[j, i, 0:4] = xywh
            target[j, i, 4] = 1
            target[j, i, 5:9] = xywh
            target[j, i, 9] = 1
        return target
