import os

import cv2
import numpy as np
from utils import get_config, get_csv_folds
from dataset.dataset_stuff import H5LikeFileInterface

from dataset.dataset_stuff import SegmentationDataset
from scipy.spatial.distance import dice
import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm


from dataset.dataset_stuff import ValDataset, SequentialDataset
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader


class Flip:
    FLIP_NONE = 0
    FLIP_LR = 1
    FLIP_FULL = 2


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    return batch.index_select(3, torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns))))).cuda())


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    return batch.index_select(2, torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows))))).cuda())


def to_numpy(batch):
    if isinstance(batch, tuple):
        batch = batch[0]
    return F.sigmoid(batch).data.cpu().numpy()


def predict(model, batch, flips=Flip.FLIP_NONE):
    pred1 = model(batch)
    if flips > Flip.FLIP_NONE:
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > Flip.FLIP_LR:
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        return to_numpy(new_mask)
    return to_numpy(pred1)


def read_model(weights_path, project, fold):
    model = nn.DataParallel(torch.load(os.path.join(weights_path, project, 'fold{}_best.pth'.format(fold))).module)
    model.eval()
    return model


def heatmap(mask):
    mask = (mask * 255).astype(np.uint8)
    return cv2.applyColorMap(mask, cv2.COLORMAP_BONE)


class Evaluator:
    def __init__(self, config, ds, folds, test=False, flips=0, num_workers=0, border=0):
        self.config = config
        self.ds = ds
        self.folds = folds
        self.test = test
        self.flips = flips
        self.num_workers = num_workers

        self.full_image = None
        self.full_mask = None
        self.current_mask = None
        self.full_pred = None
        self.border = border
        self.folder = config.folder
        self.prev_name = None
        self.on_new = False
        self.show_mask = config.dbg
        self.need_dice = False
        self.dice = []

        if self.config.save_images:
            os.makedirs(os.path.join('results', self.config.folder), exist_ok=True)

    def get_data(self, data):
        names = data['image_name']
        samples = data['image'].numpy()
        if self.need_dice or self.show_mask:
            masks = data['mask'].numpy()
            masks = np.moveaxis(masks, 1, -1)
        else:
            masks = None
        if self.config.dbg:
            samples = np.moveaxis(samples, 1, -1)
        else:
            samples = None
        return names, samples, masks

    def save(self, name, prefix=""):
        name, ext = os.path.splitext(name)
        cv2.imwrite(os.path.join('results', self.config.folder, "{}{}.png".format(prefix, name)),
                    (self.full_pred * 255).astype(np.uint8))

    def predict(self, skip_folds=None):
        # PER FOLD MODEL PREDICTION
        for fold, (train_index, val_index) in enumerate(self.folds):
            if (skip_folds is not None) and (fold in skip_folds):
                    continue
            print("fold", fold)
            prefix = ('fold' + str(fold) + "_") if self.test else ""
            self.prev_name = None
            # CREATE DATASET
            ds_class = ValDataset if not self.test else SequentialDataset
            val_dataset = ds_class(self.ds, val_index, config=self.config)
            val_dl = PytorchDataLoader(val_dataset,
                                       batch_size=self.config.predict_batch_size,
                                       num_workers=self.num_workers,
                                       drop_last=False)
            # LOAD MODEL
            weights_path = os.path.join(self.config.models_dir)
            model = read_model(weights_path, self.folder, fold)
            pbar = val_dl if self.config.dbg else tqdm.tqdm(val_dl, total=len(val_dl))
            for data in pbar:
                self.show_mask = 'mask' in data and self.show_mask
                if 'mask' not in data:
                    self.need_dice = False
                # predict with TTA
                with torch.no_grad():
                    samples = torch.autograd.Variable(data['image'].cuda(), volatile=True)
                    predicted = predict(model, samples, flips=self.flips) # predict with TTA results aggregation

                # create mask
                names, samples, masks = self.get_data(data)
                for i in range(len(names)):
                    self.prev_name = names[i]
                    self.full_pred = np.squeeze(predicted[i, ...])
                    if samples is not None:
                        self.full_image = (samples[i, ...] * 255).astype(np.uint8)
                    if masks is not None:
                        self.full_mask = (np.squeeze(masks[i, ...]) * 255).astype(np.uint8)
                    self.finalyze_mask(prefix)

                if not self.config.dbg and self.need_dice:
                    pbar.set_postfix(dice="{:.5f}".format(np.mean(self.dice)))
            if self.config.use_crop:
                self.finalyze_mask(prefix=prefix)

    def finalyze_mask(self, prefix=""):
        if self.full_mask is not None:
            if np.any(self.full_pred > .5) or np.any(self.full_mask >= 1):
                d = 1 - dice(self.full_pred.flatten() > .5, self.full_mask.flatten() >= 1)
                self.dice.append(d)
                if self.config.dbg:
                    print(self.prev_name, ' dice: ', d)
            else:
                return
        if self.config.dbg:
            self.visualize(show_light=True)
        if self.config.save_images:
            self.save(self.prev_name, prefix=prefix)

    def visualize(self, show_light=False, show_base=True):
        dsize = None
        hmap = heatmap(self.full_pred)
        if self.full_image is not None and show_light:
            light_heat = cv2.addWeighted(self.full_image[:, :,:3], 0.6, hmap, 0.4, 0)
            if dsize:
                light_heat = cv2.resize(light_heat, (dsize, dsize))
            cv2.imshow('light heat', light_heat)
            if self.full_mask is not None and self.show_mask:
                light_mask = cv2.addWeighted(self.full_image[:,:,:3], 0.6, cv2.cvtColor(self.full_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
                if dsize:
                    light_mask = cv2.resize(light_mask, (dsize, dsize))
                cv2.imshow('light mask', light_mask)
        if self.full_image is not None and show_base:
            if dsize:
                cv2.imshow('image', cv2.resize(self.full_image[:,:,:3], (dsize, dsize)))
            else:
                cv2.imshow('image', self.full_image[:,:,:3])
            if dsize:
                hmap = cv2.resize(hmap, (dsize, dsize))
            cv2.imshow('heatmap', hmap)
            if self.full_mask is not None and self.show_mask:
                if dsize:
                    cv2.imshow('mask', cv2.resize(self.full_mask, (dsize, dsize)))
                else:
                    cv2.imshow('mask', self.full_mask)
        if show_light or show_base:
            cv2.waitKey()


def eval_config(config_path):
    test = True
    config = get_config(config_path)

    num_workers = 0 if os.name == 'nt' else 3
    root = config.dataset_path
    image_folder_name = 'train' if not test else 'test'
    sds = SegmentationDataset(root,
                              config.img_rows,
                              config.img_cols,
                              image_folder_name=image_folder_name,
                              config=config,
                              apply_clahe=config.use_clahe)
    ds = H5LikeFileInterface(sds)
    folds = [([], list(range(len(sds)))) for _ in range(config.folds_num)]
    evaluator = Evaluator(config, ds, folds, test=test, flips=Flip.FLIP_LR, num_workers=num_workers, border=0)
    evaluator.need_dice = True
    print(config.folds_num)
    skip_folds = [i for i in range(config.folds_num) if config.fold is not None and i != int(config.fold)]
    print(skip_folds)
    evaluator.predict(skip_folds=skip_folds)


if __name__ == "__main__":
    for config in sorted(os.listdir('configs')):
        if not config.endswith('.json'):
            continue
        eval_config(os.path.join('configs', config))
