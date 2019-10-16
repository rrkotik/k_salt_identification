import argparse
import os
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from utils import get_config, get_csv_folds
from dataset.dataset_stuff import SegmentationDataset, H5LikeFileInterface, TrainDataset, ValDataset
from transforms import augment_airbus_base

from pytorch_zoo import linknet, unet
import time

models = {
    'linknet34': linknet.LinkNet34,
    'linkunet': linknet.LinkUNet,
    'unet': unet.Unet,
    'linkunethpc': linknet.LinkUNetHyperColumnC,
    'linkunethps': linknet.LinkUNetHyperColumnS,
    'linknetspt': linknet.LinkNet34SpatialTop,
    'linknetspd': linknet.LinkNet34SpatialDown,
    'linknetspa': linknet.LinkNet34SpatialAll,
    'linknetspt4': linknet.LinkNet34SpatialTop4,
    'linknetspd4': linknet.LinkNet34SpatialDown4,
    'linknetspa4': linknet.LinkNet34SpatialAll4,
    'linknet34heavy': linknet.LinkNet34Heavy,
    'linknet50': linknet.LinkNet50,
    'linknet50heavy': linknet.LinkNet50,
    'linknet101': linknet.LinkNet101,
    'linknet152': linknet.LinkNet152,
    'vgg19': linknet.LinkNetVGG19,
    'linknet152fpn': linknet.LinkNet152FPN,
    'linknet152skip': linknet.LinkNet152SkipConnection,
    'linknet34skip': linknet.LinkNet34Skip,
}


def target_metric(preds, trues, weight=None, is_average=True):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    preds = (preds + 0.05).round().type(torch.IntTensor).cuda()
    preds = preds.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    trues = trues.type(torch.IntTensor).cuda()
    trues = trues.squeeze(1)
    SMOOTH = 1e-6
    intersection = (preds & trues).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (preds | trues).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    thresholded = thresholded.mean()
    return thresholded


def dice_loss(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = 2. * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1)

    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


def dice_clamp(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)


class DiceLoss(nn.Module):

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        return 1 - dice_loss(F.sigmoid(input), target, weight=weight, is_average=self.size_average)


class BCEDiceLoss(nn.Module):

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)

    def forward(self, input, target, weight=None):
        return (nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average, weight=weight)(input - 0.15, target) +
                self.dice(input -0.15, target, weight=weight))


def adjust_lr(optimizer, epoch, init_lr=0.1, num_epochs_per_decay=10, lr_decay_factor=0.1):
    lr = init_lr * (lr_decay_factor ** (epoch // num_epochs_per_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cyclic_lr(optimizer, epoch, init_lr=1e-4, num_epochs_per_cycle=10, cycle_epochs_decay=4, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


losses = {
    'bce_dice_loss': BCEDiceLoss
}


class PytorchTrain:

    def __init__(self, fold, config, metrics):
        logdir = os.path.join('..', 'logs', config.folder, 'fold{}'.format(fold))
        os.makedirs(logdir, exist_ok=True)
        self.config = config
        self.fold = fold
        self.model = models[config.network](num_classes=1, num_channels=config.num_channels)
        #learnable_parameters = [param for param, _ in self.model.named_parameters()
        #                        if param not in self.model.freeze_layer_names]
        #learnable_parameters = {'params': learnable_parameters}
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        else:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=config.lr)
        self.model = nn.DataParallel(self.model).cuda()
        self.criterion = losses[config.loss]().cuda()
        self.writer = SummaryWriter(logdir)
        self.metrics = metrics
        self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        if os.name == 'nt':
            self.devices = ','.join(str(d + 5) for d in map(int, self.devices.split(',')))
        self.cache = None
        self.cached_loss = 0
        self.hard_mining = True

    def make_step(self, data, training, i):
        result = {}
        images = data['image']
        ytrues = data['mask']
        if training:
            images = Variable(images.cuda(async=True))
            ytrues = Variable(ytrues.cuda(async=True))
        else:
            images = Variable(images.cuda(async=True), volatile=True)
            ytrues = Variable(ytrues.cuda(async=True), volatile=True)
        ypreds = self.model(images)
        loss = self.criterion(ypreds, ytrues) / self.config.iter_size

        result['loss'] = loss.data
        for name, func in self.metrics:
            #acc = func(F.sigmoid(ypreds)[:, :, self.config.border:self.config.target_rows-self.config.border,
            #                                   self.config.border:self.config.target_cols-self.config.border].contiguous(),
            #                             ytrues[:, :, self.config.border:self.config.target_rows-self.config.border,
            #                                          self.config.border:self.config.target_cols-self.config.border].contiguous())
            acc = func(F.sigmoid(ypreds).contiguous(),
                       ytrues.contiguous())
            result[name] = acc.data

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1. * self.config.iter_size)
            if i % self.config.iter_size == 0:
                self.optimizer.step()
        if training and i % self.config.iter_size == 0:
            self.optimizer.zero_grad()
        return result

    def run_one_epoch(self, epoch, loader, training=True):
        avg_meter = defaultdict(float)

        if training:
            pbar = tqdm(enumerate(loader), total=len(loader), desc="Epoch {}".format(epoch), ncols=0)
        else:
            pbar = enumerate(loader)
        for i, data in pbar:
            meter = self.make_step(data, training, i)
            for k, val in meter.items():
                avg_meter[k] += val
                if training and self.hard_mining and k == 'loss':
                    loss = val.cpu().numpy()
                    if self.cache is None or self.cached_loss < loss:
                        self.cached_loss = loss
                        self.cache = deepcopy(data)

            if training:
                if self.hard_mining and i % 10 == 0 and self.cache is not None:
                    self.make_step(self.cache, training, i)
                    self.cache = None
                    self.cached_loss = 0
                if i % 100 == 0 or i == len(loader) - 1:
                    print({k: "{:.5f}".format(v.cpu().numpy() / (i + 1)) for k, v in avg_meter.items()})
        return {k: v.cpu().numpy() / len(loader) for k, v in avg_meter.items()}

    def fit(self, train_loader, val_loader):
        save_path = os.path.join(self.config.models_dir, self.config.folder)
        os.makedirs(save_path, exist_ok=True)
        best_epoch = -1
        best_loss = float('inf')
        best_soft_dice = float('inf')

        for epoch in range(self.config.nb_epoch):
            start_time = time.time()
            lr = self.optimizer.param_groups[0]['lr']
            #if epoch == self.config.warmup_epoch:
            #    self.optimizer.add_param_group({'params': self.model.freeze_layer_names})
            if epoch == self.config.cycle_start_epoch:
                print("Starting cyclic lr")
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            if epoch >= self.config.cycle_start_epoch:
                lr = cyclic_lr(self.optimizer, epoch - self.config.cycle_start_epoch,
                               init_lr=self.config.lr, num_epochs_per_cycle=5, cycle_epochs_decay=4,
                               lr_decay_factor=0.1)
            elif epoch >= self.config.warmup_epoch:
                lr = adjust_lr(self.optimizer, epoch - self.config.warmup_epoch,
                               init_lr=self.config.lr,
                               num_epochs_per_decay=self.config.lr_decay_epoch_num)
            self.optimizer.zero_grad()
            self.model.train()
            train_metrics = self.run_one_epoch(epoch, train_loader)
            self.model.eval()
            val_metrics = self.run_one_epoch(epoch, val_loader, training=False)
            print(" | ".join("{}: {:.5f}".format(k, float(v)) for k, v in val_metrics.items()))

            for k, v in train_metrics.items():
                self.writer.add_scalar('train/{}'.format(k), float(v), global_step=epoch)

            for k, v in val_metrics.items():
                self.writer.add_scalar('val/{}'.format(k), float(v), global_step=epoch)
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('lr', float(lr), global_step=epoch)
            loss = -float(val_metrics['soft dice'])
            if -float(val_metrics['soft dice']) < best_soft_dice:
                best_soft_dice = -float(val_metrics['soft dice'])
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(deepcopy(self.model), os.path.join(save_path, 'fold{}_best.pth'.format(self.fold)))
            print("Epoch No {} took {}".format(epoch, time.time() - start_time))
            self.writer.add_scalar('lr', float(lr), global_step=epoch)
        with open(os.path.join(save_path, 'scores.csv'), 'a') as outf:
            outf.write("target,{},{}\n".format(best_loss,best_soft_dice))
        print('Finished {} fold: {} with best loss {:.5f} on epoch {}'.format(self.config.folder, self.fold,
                                                                              best_loss, best_epoch))
        self.writer.close()


def train(ds, folds, config, num_workers=0, transforms=None, skip_folds=None):
    os.makedirs(os.path.join('..', 'weights'), exist_ok=True)
    os.makedirs(os.path.join('..', 'logs'), exist_ok=True)
    save_path = os.path.join(config.models_dir, config.folder)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'scores.csv'), 'w') as outf:
        outf.write('metric,score\n')

    for fold, (train_idx, val_idx) in enumerate(folds):
        if skip_folds and fold in skip_folds:
            print("skipping fold ", fold)
            continue
        tr = TrainDataset(ds, train_idx, config, transform=transforms)
        val = ValDataset(ds, val_idx, config, transform=None)
        train_loader = PytorchDataLoader(tr,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=num_workers,
                                         pin_memory=True)

        val_loader = PytorchDataLoader(val,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=num_workers,
                                       pin_memory=True)
        trainer = PytorchTrain(fold=fold,
                               config=config,
                               metrics=[('soft dice', dice_loss),
                                        ('hard dice', dice_clamp),
                                        ('bce', nn.modules.loss.BCELoss()),
                                        ('target', target_metric)])
        trainer.fit(train_loader, val_loader)
        trainer.writer.close()


def train_config(config_path):
    config = get_config(config_path)

    root = config.dataset_path
    image_folder_name = 'train'
    ds = H5LikeFileInterface(SegmentationDataset(root,
                                                 config.img_rows, config.img_cols,
                                                 image_folder_name=image_folder_name,
                                                 config=config,
                                                 apply_clahe=config.use_clahe))

    f = config.split_name
    print("Config split name ", f)
    folds = get_csv_folds(os.path.join('configs', f + '.csv'))
    num_workers = 8 if os.name == 'nt' else 5

    skip_folds = [i for i in range(config.folds_num) if config.fold is not None and i != int(config.fold)]
    print('skipping folds: ', skip_folds)
    #skip_folds = []
    train(ds, folds, config, num_workers=num_workers, transforms=globals()[config.augmentation], skip_folds=skip_folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='segmentation train script')
    parser.add_argument('-c', '--config', help='config file', default=None)
    for config in sorted(os.listdir('configs')):
        if not config.endswith('.json'):
            continue
        print(config)
        train_config(os.path.join('configs', config))
