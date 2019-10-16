import argparse
import cv2
import os
from os.path import join, split, splitext
import numpy as np
from multiprocessing import Pool, freeze_support
from functools import partial
import json

THRESHOLD = 128
BORDER = 12


def RLenc(img, order='F'):
    """Convert binary mask image to run-length array or string.

    Args:
    img: image in shape [n, m]
    order: is down-then-right, i.e. Fortran(F)
    string: return in string or array

    Return:
    run-length as a string: <start[1s] length[1s] ... ...>
    """
    img = img > 0
    bytez = img.reshape(img.shape[0] * img.shape[1], order=order)
    bytez = np.concatenate([[0], bytez, [0]])
    runs = np.where(bytez[1:] != bytez[:-1])[0] + 1  # pos start at 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class EvalAggregator:

    def __init__(self, config_name, binarization_threshold):
        with open(os.path.join('configs', config_name), 'r') as f:
            self.config = json.load(f)
            self.binarization_threshold = binarization_threshold

    def postprocess_image(self, image):
        border_pad_margin = self.config['border'] + self.config['padding']
        #image = cv2.resize(image, (self.config['submission_cols'] + border_pad_margin * 2,
        #                           self.config['submission_rows'] + border_pad_margin * 2))
        image = image[border_pad_margin: image.shape[0] - border_pad_margin,
                      border_pad_margin: image.shape[1] - border_pad_margin]
        image = image[:, 8:-8]
        #return cv2.resize(image, (self.config['submission_cols'], self.config['submission_rows']))
        return image

    def output_image(self, output_path, fname, image):
        image = self.postprocess_image(image)
        if output_path.endswith('.csv'):  # make rle
            # make rle and dump to csv
            res_str = RLenc(image)
            fname = splitext(split(fname)[1])[0]
            return '{},{}\n'.format(fname, res_str) if len(res_str) > 0 else '{},\n'.format(fname)
            #with open(output_path, 'a') as outf:
            #    outf.write('{}, {}\n'.format(fname, res_str))
        #else:
        # cv2.imwrite(os.path.join(submissions_dir,
        #                         'ln34_{}'.format(binarization),
        #                         prob_file.replace('_sat', '_mask')),
        #            mean)


    def process_1image(self, fname, roots, output_path, binarization=THRESHOLD):
        ims = []
        for fold in range(self.config["folds_num"]):
            for r in roots:
                pred_path = os.path.join(r, 'fold{}_'.format(fold) + fname)
                im = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                ims.append(im)
        mean = (np.mean(ims, axis=0)).astype(np.uint8)
        # mean = cv2.cvtColor(mean, cv2.COLOR_GRAY2BGR)
        mean[mean <= binarization] = 0
        mean[mean > binarization] = 255
        return self.output_image(output_path, fname, mean)



    def run(self):
        roots = [
            os.path.join(r'results', self.config["folder"]),
        ]
        submissions_dir = join('results', 'final_' + self.config["folder"])
        os.makedirs(submissions_dir, exist_ok=True)
        predicted_folded_images = os.listdir(roots[0])
        image_names = {f[6:] for f in predicted_folded_images if f.startswith('fold') if f.endswith('.png')}
        output_file = os.path.join(submissions_dir, '{}_{}.csv'.format(self.config['folder'], self.binarization_threshold))
        #f = partial(folds_mean,
        #            folds_num=config['folds_num'],
        #            roots=roots,
        #            output_path=output_file,
        #            binarization=threshold)
        #with Pool() as pool:
        #    encoded_images = pool.map(f, image_names)
        encoded_images = []
        for iidx, imname in enumerate(image_names):
            if iidx % 1000 == 0:
                print(iidx)
            encoded_images.append(self.process_1image(imname,
                                                      roots=roots,
                                                      output_path=output_file,
                                                      binarization=self.binarization_threshold))
        with open(os.path.join(submissions_dir, '{}_{}.csv'.format(self.config['folder'], self.binarization_threshold)), 'w') as out_file:
            out_file.write('id,rle_mask\n')
            for enc_img in encoded_images:
                out_file.write(enc_img)


def get_parser():
    parser_ = argparse.ArgumentParser(description="Final evaluation results aggregation")
    parser_.add_argument('-c', '--config', help="config filename", default='default.json')
    parser_.add_argument('-th', '--threshold', help="binarization threshold", default=128, type=int)
    return parser_


if __name__ == "__main__":
    freeze_support()
    parser = get_parser()
    args = parser.parse_args()
    eval_aggregator = EvalAggregator(args.config, args.threshold)
    eval_aggregator.run()
