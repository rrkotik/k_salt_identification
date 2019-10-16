import numpy as np
from collections import defaultdict, OrderedDict
import os
from PIL import Image
import json
import pandas as pd


def generate_area_folds():
    with open(os.path.join('configs', 'default.json'), 'r') as f:
        config = json.load(f)
    TRAIN_DATA = os.path.join('data', 'train_masks')
    train_files = os.listdir(TRAIN_DATA)
    train_ids = {s.split('.png')[0] for s in train_files}

    squares_fma = {}
    for idx in train_ids:
        im = np.asarray(Image.open(
                 os.path.join(TRAIN_DATA,
                             idx + ".png")))
        fma = np.mean(im)
        squares_fma[idx] = fma

    squares_fma = OrderedDict(sorted(squares_fma.items(), key=lambda x: x[1]))
    fma_rows = [(i, square * 100, fold % config['folds_num'])
                for fold, (i, square) in enumerate(squares_fma.items())]

    fma_df = pd.DataFrame(fma_rows, columns=['id', 'square', 'fold'])

    fma_df.to_csv('configs/area_folds.csv', index=False)


def generate_area_depth_folds():
    with open(os.path.join('configs', 'default.json'), 'r') as f:
        config = json.load(f)
    TRAIN_DATA = os.path.join(config['dataset_path'], 'train/masks')
    depth_data = pd.read_csv(os.path.join(config['dataset_path'], 'depths.csv'))
    train_files = os.listdir(TRAIN_DATA)
    train_ids = list({s.split('.png')[0] for s in train_files})
    squares_fma = []
    depths = []
    for idx in train_ids:
        im = np.asarray(Image.open(
                 os.path.join(TRAIN_DATA,
                             idx + ".png")))
        fma = np.mean(im)
        squares_fma.append(fma)
        depths.append(depth_data['z'][depth_data['id'] == idx].values[0])
    df = pd.DataFrame({'id' : train_ids, 'square' : squares_fma, 'depth' : depths})
    df = df[['id', 'depth', 'square']]
    df = df.sort_values(['depth', 'square'], ascending=[True, True])
    folds = []
    for idx in range(len(df)):
        folds.append(idx % config['folds_num'])
    df['fold'] = pd.Series(folds, index=df.index)
    df.to_csv('configs/depth_area_folds.csv', index=False)



def generate10rndfolds(seed=13):
    np.random.seed(seed)
    TRAIN_DATA = os.path.join('data', 'train/masks')
    train_files = os.listdir(TRAIN_DATA)
    train_ids = list({s.split('.png')[0] for s in train_files})
    np.random.shuffle(train_ids)
    df = pd.DataFrame({'id' : train_ids, 'square' : [-1] * len(train_ids), 'depth' : [-1] * len(train_ids)})
    df = df[['id', 'depth', 'square']]
    folds = []
    for idx in range(len(df)):
        folds.append(idx % 10)
    df['fold'] = pd.Series(folds, index=df.index)
    df.to_csv('configs/10random.csv', index=False)



if __name__ == "__main__":
    generate_area_folds()
    # generate_area_depth_folds()
    #generate10rndfolds()
