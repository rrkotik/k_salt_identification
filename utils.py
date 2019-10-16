import numpy as np
from config import Config
import os
import json
import pandas as pd


def get_csv_folds(path):
    df = pd.read_csv(path)
    df = df[['id', 'fold']]

    folds_num = len(np.unique(df['fold'].values))

    train = [[] for _ in range(folds_num)]
    test = [[] for _ in range(folds_num)]

    folds = {}
    for i in range(folds_num):
        fold_ids = list(df[df['fold'].isin([i])].index.values)
        fold_ids = [int(fid) for fid in fold_ids]
        folds.update({i: fold_ids})

    for k, v in folds.items():
        for i in range(folds_num):
            if i != k:
                train[i].extend(v)
        test[k] = v
    return list(zip(np.array(train, dtype='object'), np.array(test, dtype='object')))


def get_config(config_path):
    print(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        config['fold'] = None
    if 'lr_decay_epoch_num' not in config:
        config['lr_decay_epoch_num'] = 10
    return Config(**config)
