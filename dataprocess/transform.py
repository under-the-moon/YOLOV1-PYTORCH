import cv2
import numpy as np


class Transform:

    def __init__(self, rand=.5):
        self.rand = rand
        self.img_ts = [Blur(), Hue(), Saturation()]
        self.ts = [Scale(), Flip(), Crop(), Translate(), Identify()]
        self.mix_ts = [MixUp(), Mosaic()]

    def __call__(self, data):
        img = data['img']
        boxes = data['boxes']

        if np.random.rand() < self.rand:
            img_t = np.random.choice(self.img_ts)
            img = img_t(img)

            t = np.random.choice(self.ts)
            tmp_data = {'img': img, 'boxes': boxes}
            new_data = t(tmp_data)

            if np.random.rand() < self.rand:
                mix_t = np.random.choice(self.mix_ts)
                new_data = mix_t(new_data)
            return new_data
        return data


class Blur:
    def __init__(self, size=(5, 5)):
        self.size = size

    def __call__(self, img):
        return cv2.blur(img, self.size)


class Hue:
    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        adjust = np.random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img


class Saturation:
    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        adjust = np.random.choice([0.5, 1.5])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img


class Brightness:
    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        adjust = np.random.choice([0.5, 1.5])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img


class Identify:
    def __call__(self, data):
        return data


class Scale:
    def __init__(self, min=.8, max=1.2):
        self.min = min
        self.max = max

    def __call__(self, data):
        img = data['img']
        boxes = data['boxes']
        h, w = img.shape[0:2]
        scale = np.random.uniform(self.min, self.max)
        nh = h
        nw = int(w * scale)
        boxes[:, 0:4] = boxes[:, 0:4] * np.array([[scale, 1, scale, 1]])

        img = cv2.resize(img, (nw, nh))
        return {'img': img, 'boxes': boxes}


class Flip:
    FLIP_TOP_BOTTOM = 0
    FLIP_LEFT_RIGHT = 1
    FLIP_ALL_DIRECTION = -1

    def __init__(self):
        self.flips = [self.FLIP_TOP_BOTTOM, self.FLIP_LEFT_RIGHT, self.FLIP_ALL_DIRECTION]

    def __call__(self, data):
        img = data['img']
        boxes = data['boxes']
        h, w = img.shape[0:2]
        flip = np.random.choice(self.flips)
        img = cv2.flip(img, flip)
        if flip == self.FLIP_LEFT_RIGHT:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        elif flip == self.FLIP_TOP_BOTTOM:
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        else:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return {'img': img, 'boxes': boxes}


class Crop:
    def __call__(self, data):
        img = data['img']
        boxes = data['boxes']

        h, w = img.shape[0:2]
        max_bbox = np.concatenate(
            [
                np.min(boxes[:, 0:2], axis=0),
                np.max(boxes[:, 2:4], axis=0),
            ],
            axis=-1,
        )
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]
        crop_xmin = max(
            0, int(max_bbox[0] - np.random.uniform(0, max_l_trans))
        )
        crop_ymin = max(
            0, int(max_bbox[1] - np.random.uniform(0, max_u_trans))
        )
        crop_xmax = min(
            w, int(max_bbox[2] + np.random.uniform(0, max_r_trans))
        )
        crop_ymax = min(
            h, int(max_bbox[3] + np.random.uniform(0, max_d_trans))
        )

        img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_xmin
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_ymin

        return {'img': img, 'boxes': boxes}


class Translate:

    def __call__(self, data):
        img = data['img']
        boxes = data['boxes']
        pre_boxes = np.copy(boxes)

        h, w = img.shape[0:2]
        # 最大平移四分之一
        tx = np.random.randint(0, w // 4)
        ty = np.random.randint(0, h // 4)
        # # 是否进行反向平移
        reverse = True if np.random.rand() > .5 else False
        if reverse:
            tx = -tx
            ty = - ty
        new_img = np.empty_like(img)
        new_img.fill(0)
        boxes[:, [0, 2]] += tx
        boxes[:, [1, 3]] += ty
        new_boxes = []
        if not reverse:
            new_img[ty:, tx:, :] = img[:h - ty, :w - tx, :]
            for box in boxes:
                x1, y1, x2, y2, cls = box
                if x1 >= w or y1 >= h:
                    continue
                x2 = w - 1 if x2 >= w else x2
                y2 = h - 1 if y2 >= h else y2
                new_boxes.append([x1, y1, x2, y2, cls])
        else:
            new_img[:h + ty, :w + tx, :] = img[-ty:, -tx:, :]
            for box in boxes:
                x1, y1, x2, y2, cls = box
                if x2 <= 0 or y2 <= 0:
                    continue
                x1 = 1 if x1 <= 0 else x1
                y1 = 1 if y1 <= 0 else y1
                new_boxes.append([x1, y1, x2, y2, cls])
        if len(new_boxes) == 0:
            data['boxes'] = pre_boxes
            return data
        new_boxes = np.array(new_boxes)
        return {'img': new_img, 'boxes': new_boxes}


class MixUp:
    def __call__(self, data):
        img1 = data['img']
        boxes1 = data['boxes']
        img2 = np.copy(img1)
        boxes2 = np.copy(boxes1)

        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]

        nh, nw = max(h1, h2), max(w1, w2)

        new_img = np.empty((nh, nw, 3), dtype=np.uint8)
        new_img.fill(0)

        random_w = nw // 5
        cutx = np.random.randint(random_w * 2, random_w * 3)

        nw1 = cutx
        nw2 = nw - cutx

        img1 = cv2.resize(img1, (nw1, nh))
        img2 = cv2.resize(img2, (nw2, nh))
        new_img[:, 0:cutx, :] = img1
        new_img[:, cutx:, :] = img2
        boxes1[:, [0, 2]] = boxes1[:, [0, 2]] * nw1 / w1
        boxes1[:, [1, 3]] = boxes1[:, [1, 3]] * nh / h1
        boxes2[:, [0, 2]] = boxes2[:, [0, 2]] * nw2 / w2 + cutx
        boxes2[:, [1, 3]] = boxes2[:, [1, 3]] * nh / h2
        new_boxes = np.concatenate([boxes1, boxes2], axis=0)
        return {'img': new_img, 'boxes': new_boxes}


class Mosaic:
    def __call__(self, data):
        img1 = data['img']
        boxes1 = data['boxes']
        img2 = np.copy(img1)
        boxes2 = np.copy(boxes1)
        img3 = np.copy(img1)
        boxes3 = np.copy(boxes1)
        img4 = np.copy(img1)
        boxes4 = np.copy(boxes1)

        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]
        h3, w3 = img3.shape[0:2]
        h4, w4 = img4.shape[0:2]

        nh, nw = max(h1, h2, h3, h4), max(w1, w2, w3, w4)

        random_w = nw // 5
        cutx = np.random.randint(random_w * 2, random_w * 3)

        random_h = nh // 5
        cuty = np.random.randint(random_h * 2, random_h * 3)

        nw1, nh1 = cutx, cuty
        nw2, nh2 = nw - cutx, cuty
        nw3, nh3 = cutx, nh - cuty
        nw4, nh4 = nw - cutx, nh - cuty

        img1 = cv2.resize(img1, (nw1, nh1))
        img2 = cv2.resize(img2, (nw2, nh2))
        img3 = cv2.resize(img3, (nw3, nh3))
        img4 = cv2.resize(img4, (nw4, nh4))

        new_img = np.empty((nh, nw, 3), dtype=np.uint8)
        new_img.fill(0)
        new_img[:cuty, :cutx, :] = img1
        new_img[:cuty, cutx:, :] = img2
        new_img[cuty:, :cutx, :] = img3
        new_img[cuty:, cutx:, :] = img4

        boxes1[:, [0, 2]] = boxes1[:, [0, 2]] * nw1 / w1
        boxes1[:, [1, 3]] = boxes1[:, [1, 3]] * nh1 / h1

        boxes2[:, [0, 2]] = boxes2[:, [0, 2]] * nw2 / w2 + cutx
        boxes2[:, [1, 3]] = boxes2[:, [1, 3]] * nh2 / h2

        boxes3[:, [0, 2]] = boxes3[:, [0, 2]] * nw3 / w3
        boxes3[:, [1, 3]] = boxes3[:, [1, 3]] * nh3 / h3 + cuty

        boxes4[:, [0, 2]] = boxes4[:, [0, 2]] * nw4 / w4 + cutx
        boxes4[:, [1, 3]] = boxes4[:, [1, 3]] * nh4 / h4 + cuty

        new_boxes = np.concatenate([boxes1, boxes2, boxes3, boxes4], axis=0)

        return {'img': new_img, 'boxes': new_boxes}