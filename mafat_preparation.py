import numpy as np
import cv2

ENLARGE = 480


def enlarge_image(image):
    shapes = [sz + ENLARGE * 2 for sz in image.shape[:2]]
    enlarged = np.zeros(shapes[:2] + image.shape[2:], dtype=image.dtype)
    enlarged[ENLARGE: shapes[0] - ENLARGE, ENLARGE: shapes[1] - ENLARGE] = image
    return enlarged


def enlarge_bbox(bbox, scale=1.5):
    assert scale > 1.
    x0, y0, x1, y1, x2, y2, x3, y3 = bbox
    x_min = min([x0, x1, x2, x3])
    x_max = max([x0, x1, x2, x3])
    y_min = min([y0, y1, y2, y3])
    y_max = max([y0, y1, y2, y3])
    x_center = (x_min + x_max) / 2.
    y_center = (y_min + y_max) / 2.
    x_size = x_max - x_min
    y_size = y_max - y_min
    x_enlarge = int(x_size * (scale - 1) / 2.)
    y_enlarge = int(y_size * (scale - 1) / 2.)
    newbbox = []
    for coord_idx, coord in enumerate(bbox): # bbox should be x0, y0, x1, y1, x2, y2, x3, y3
        if coord_idx % 2 == 0: #this is x
            if coord < x_center:
                newbbox.append(int(coord - x_enlarge))
            else:
                newbbox.append(int(coord + x_enlarge))
        else:
            if coord < y_center:
                newbbox.append(int(coord - y_enlarge))
            else:
                newbbox.append(int(coord + y_enlarge))
    return newbbox


def crop_bbox(image, bbox):
    # bbox to cnt
    cnt = [(bbox[idx], bbox[idx] + 1) for idx in range(0, len(bbox), 2)]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2 - x1, y2 - y1)
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))
    return croppedRotated