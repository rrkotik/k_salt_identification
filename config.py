from collections import namedtuple

Config = namedtuple("Config", [
    "dataset_path",
    "models_dir",
    "folder",
    "img_rows",
    "img_cols",
    "target_rows",
    "target_cols",
    "num_channels",
    "network",
    "loss",
    "lr",
    "optimizer",
    "batch_size",
    "epoch_size",
    "use_clahe",
    "nb_epoch",
    "cycle_start_epoch",
    "predict_batch_size",
    "use_crop",
    "use_resize",
    "dbg",
    "save_images",
    "padding",
    "fold",
    "folds_num",
    "split_name",
    "iter_size",
    "border",
    "submission_cols",
    "submission_rows",
    "augmentation",
    "lr_decay_epoch_num",
    "warmup_epoch"
])


