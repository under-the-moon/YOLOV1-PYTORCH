import matplotlib.pyplot as plt
import numpy as np
import cv2


# def plt_show(image, boxes):
#     image = image.astype(np.uint8)
#     plt.figure(figsize=(7, 7))
#     plt.axis("off")
#     plt.imshow(image)
#     ax = plt.gca()
#     for box in boxes:
#         x1, y1, x2, y2, cls = [int(cor) for cor in box]
#         w, h = x2 - x1, y2 - y1
#         patch = plt.Rectangle(
#             [x1, y1], w, h, fill=False, edgecolor=[1, 0, 0], linewidth=1
#         )
#         ax.add_patch(patch)
#     plt.show()
#
#
# def cv2_show(image, boxes):
#     image = image.astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     for box in boxes:
#         x1, y1, x2, y2, cls = [int(cor) for cor in box]
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
#     cv2.imshow('image', image)
#     cv2.waitKey(0)

def save_eval_img(image, boxes, save_path):
    n = boxes.shape[0]
    image = image * 255
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if n == 0:
        cv2.imwrite(save_path, image)
    else:
        for i in range(n):
            box = boxes[i]
            x1, y1, x2, y2 = [int(cor) for cor in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(save_path, image)


def save_img(image, boxes, scores, class_ids, names, colors, save_path):
    n = boxes.shape[0]
    image = image.astype(np.uint8)
    if n == 0:
        cv2.imwrite(save_path, image)
    else:
        for i in range(n):
            box = boxes[i]
            x1, y1, x2, y2 = [int(cor) for cor in box]
            cls = class_ids[i]
            score = scores[i]
            score = round(score, 2)
            name = names[cls]
            color = colors[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f'{name}: {score}'
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_ITALIC, .5, color, 2)
        cv2.imwrite(save_path, image)


def show_img(image, boxes, scores, class_ids, names, colors):
    n = boxes.shape[0]
    image = image.astype(np.uint8)
    if n == 0:
        cv2.imshow('image', image)
        cv2.waitKey(0)
    else:
        for i in range(n):
            box = boxes[i]
            x1, y1, x2, y2 = [int(cor) for cor in box]
            cls = class_ids[i]
            score = scores[i]
            score = round(score, 2)
            name = names[cls]
            color = colors[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f'{name}: {score}'
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_ITALIC, .5, color, 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
