import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from transforms import CLAHE
from torch.utils.data.dataset import Dataset as TorchDataset
import scipy.ndimage

from .crops import ImageCropper


class SegmentationDataset:
    def __init__(self, root_path, rows, cols, channels=3,
                 image_folder_name='train/images', config=None,
                 apply_clahe=False):
        self.rows = rows
        self.cols = cols
        self.final_rows = self.rows + 2 * config.padding
        self.final_cols = self.rows + 2 * config.padding
        self.channels = channels
        self.root_path = root_path
        self.im_names = []
        self.with_alpha = None
        self.image_folder_name = image_folder_name
        if 'train' in image_folder_name:
            self.im_names = pd.read_csv('configs/{}.csv'.format(config.split_name))['id'].values
            self.im_names = [str(fname) + '.jpg' for fname in self.im_names]
        else:
            self.im_names = os.listdir(os.path.join(root_path, image_folder_name))
        self.images = {}
        self.masks = {}
        self.clahe = CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.apply_clahe = apply_clahe
        self.config = config

    def reflect_border(self, image):
        return cv2.copyMakeBorder(image,
                                  self.config.border,
                                  self.config.border,
                                  self.config.border,
                                  self.config.border,
                                  cv2.BORDER_REFLECT)

    def resize_image(self, image):
        image = cv2.resize(image, (self.cols, self.rows))
        return image

    def pad_image(self, image):
        channels = image.shape[2] if len(image.shape) > 2 else None
        two_pads = 2 * self.config.padding
        shape = (image.shape[0] + two_pads, image.shape[1] + two_pads) + \
                ((image.shape[2], ) if channels else ())
        empty_image = np.zeros(shape, dtype=image.dtype)
        empty_image[self.config.padding : image.shape[0] + self.config.padding,
                self.config.padding : image.shape[1] + self.config.padding, ...] = image
        return empty_image

    def read_image(self, fname):
        im = cv2.imread(os.path.join(self.root_path, self.image_folder_name, fname))
        return self.clahe(im) if self.apply_clahe else im

    def read_mask(self, fname):
        path = os.path.join(self.root_path,
                            self.image_folder_name.replace('train', 'train_masks'),
                            fname.replace('.jpg', '.png'))
        mask = np.copy(np.asarray(Image.open(path).convert(mode='L')))
        if np.max(mask) < 255:
            mask[mask > 0] = 128
        return mask.astype(np.uint8)

    def get_image(self, idx):
        fname = self.im_names[idx]
        data = self.read_image(fname)
        return self.finalyze(data)

    def get_mask(self, idx):
        fname = self.im_names[idx]
        data = self.read_mask(fname)
        return self.finalyze(data)

    def finalyze(self, data):
        final = self.pad_image(self.reflect_border(self.resize_image(data)))
        return final

    def __len__(self):
        return len(self.im_names)


class H5LikeFileInterface:
    def __init__(self, dataset: SegmentationDataset):
        """
        :param dataset:
        """
        self.dataset = dataset
        self.current_kind = None

    def __getitem__(self, item):
        if isinstance(item, str):
            if ('masks' in item) or ('images' in item) or ('names' in item):
                self.current_kind = item
                return self
            else:
                idx = item
        elif isinstance(item, int):
            idx = item
            s = None
        elif isinstance(item, tuple):
            idx = item[0]
            s = item[1:]
        else:
            raise Exception()

        if self.current_kind == 'images':
            data = self.dataset.get_image(idx)
        elif self.current_kind == 'masks':
            data = self.dataset.get_mask(idx)
        elif self.current_kind == 'names':
            data = self.dataset.im_names[idx]
        elif self.current_kind == 'alphas':
            data = self.dataset.get_alpha(idx)
        else:
            raise Exception()
        return data[s] if s is not None else data

    def __contains__(self, item):
        return item in ['images', 'masks', 'names']

    def __len__(self):
        return len(self.dataset)


class MaskDataset(TorchDataset):
    def __init__(self, h5dataset, image_indexes, config, transform=None):
        self.cropper = ImageCropper(h5dataset.dataset.final_rows,
                                    h5dataset.dataset.final_cols,
                                    config.target_rows,
                                    config.target_cols,
                                    config.padding,
                                    config.use_crop,
                                    config.use_resize)
        self.dataset = h5dataset
        self.image_indexes = image_indexes if isinstance(image_indexes, list) else image_indexes.tolist()
        self.transform = transform
        self.config = config
        self.keys = {'image', 'mask', 'image_name'}

    def __getitem__(self, item):
        raise NotImplementedError

    def image_to_float(self, image):
        im = image
        im = im[:, :, ::-1].copy().astype(np.float32) / 255.
        im -= np.array([0.485, 0.456, 0.406])
        im /= np.array([0.229, 0.224, 0.225])
        image = im
        return np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32)

    def expand_mask(self, mask):
        return np.expand_dims(mask / 255., axis=0).astype(np.float32)

    def distance_transform(self, mask):
        emask = -scipy.ndimage.distance_transform_edt(~mask) + scipy.ndimage.distance_transform_edt(mask)
        emask_img = np.sign(emask) * np.log1p(np.abs(emask))
        return np.expand_dims(emask_img, axis=0).astype(np.float32)


class TrainDataset(MaskDataset):
    def __init__(self, h5dataset, image_indexes, config, transform=None):
        super(TrainDataset, self).__init__(h5dataset, image_indexes, config, transform)
        self.image_indexes = image_indexes

    def __getitem__(self, idx):
        """
        idx seems to be unused
        """
        im_idx = idx % len(self.image_indexes)
        im_idx = int(self.image_indexes[im_idx])
        sx, sy = self.cropper.randomCropCoords()
        name = self.dataset['names'][im_idx]
        mask = self.cropper.getImage(self.dataset, 'masks', im_idx, sx, sy)
        im = self.cropper.getImage(self.dataset, 'images', im_idx, sx, sy)
        if self.transform is not None:
            im, mask = self.transform(im, mask)
        return {'image': self.image_to_float(im),
                'mask': self.expand_mask(mask),
                'image_name': name}

    def __len__(self):
        return len(self.image_indexes) * self.config.epoch_size


class SequentialDataset(MaskDataset):
    def __init__(self, h5dataset, image_indexes, config, transform=None):
        super(SequentialDataset, self).__init__(h5dataset,
                                                image_indexes,
                                                config,
                                                transform)
        self.good_tiles = []
        self.keys = {'image', 'image_name', 'sy', 'sx'}
        self.init_good_tiles()

    def init_good_tiles(self):
        self.good_tiles = []
        positions = self.cropper.positions
        for im_idx in self.image_indexes:
            for pos in positions:
                self.good_tiles.append((im_idx, *pos))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            print("return none")
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        im = self.cropper.getImage(self.dataset, 'images', im_idx, sx, sy)
        if self.transform is not None:
            im = self.transform(im)
        name = self.dataset['names'][im_idx]
        return {'image': self.image_to_float(im),
                'startx': sx,
                'starty': sy,
                'image_name': name}

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset, MaskDataset):
    def __init__(self, h5dataset, image_indexes, config, transform=None):
        self.image_indexes = np.array(image_indexes)
        super(ValDataset, self).__init__(h5dataset,
                                         self.image_indexes,
                                         config,
                                         transform)
        self.keys = {'image', 'mask', 'image_name', 'sx', 'sy'}

    def __getitem__(self, idx):
        res = SequentialDataset.__getitem__(self, idx)
        if res is None:
            return res
        im_idx, sx, sy = self.good_tiles[idx]
        mask = self.cropper.getImage(self.dataset, 'masks', im_idx, sx, sy)
        res['mask'] = self.expand_mask(mask)
        return res

