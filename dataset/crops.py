import random
import numpy as np


class ImageCropper:
    def __init__(self, img_rows, img_cols, target_rows, target_cols, pad, use_crop, use_resize):
        self.image_rows = img_rows
        self.image_cols = img_cols
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.pad = pad
        self.use_crop = use_crop
        self.use_resize = use_resize
        self.starts_y = self.sequentialStarts(axis=0) if self.use_crop else [0]
        self.starts_x = self.sequentialStarts(axis=1) if self.use_crop else [0]
        self.positions = [(x, y) for x in self.starts_x for y in self.starts_y]

    def randomCropCoords(self):
        x = 0#random.randint(0, self.image_cols - self.target_cols)
        y = 0#random.randint(0, self.image_rows - self.target_rows)
        return x, y

    def getImage(self, data, kind, im_idx, x, y):
        d = data[kind]
        return d[im_idx, y: y + self.target_rows, x: x + self.target_cols] if self.use_crop else d[im_idx]

    def sequentialStarts(self, axis=0):
        best_dist = float('inf')
        best_starts = None
        big_segment = self.image_cols if axis == 1 else self.image_rows
        small_segment = self.target_cols if axis == 1 else self.target_rows
        opt_val = len(np.arange(0, big_segment, small_segment - self.pad)) - 1
        for i in range(small_segment - self.pad):
            r = np.arange(0, big_segment, small_segment - self.pad - i)
            minval = abs(big_segment - small_segment - r[opt_val])
            if minval < best_dist:
                best_dist = minval
                best_starts = r
            else:
                starts = best_starts[:opt_val].tolist() + [big_segment - small_segment]
                return starts

    def sequentialCrops(self, img):
        for startx in self.starts_x:
            for starty in self.starts_y:
                yield img[starty: starty + self.target_rows, startx: startx+self.target_cols, ...]
