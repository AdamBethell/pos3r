# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from pos3r.model_dust3r import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from pos3r.model import RayCroCoNet
from pos3r.datasets import get_data_loader  # noqa
from pos3r.losses import *  # noqa: F401, needed when loading the model
from pos3r.inference import loss_of_one_batch  # noqa
from pos3r.eval_utils import pose_eval, pt_eval

import pos3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('POS3R testing', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")

    # others
    parser.add_argument('--num_workers', default=8, type=int)

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser


def evaluate(args):
    print(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # training dataset and loader
    print('Building test dataset {:s}'.format(args.test_dataset))
    # data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
    #                     for dataset in args.test_dataset.split('+')}
    data_loader_test = build_dataset(args.test_dataset, args.batch_size, args.num_workers, test=True)

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    # following timm: set wd as 0 for bias and norm layers

    def write_log_stats(test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            with open(os.path.join(args.output_dir, "results.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(test_stats) + "\n")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start Evaluation")
    start_time = time.time()
    stats = test(model, data_loader_test, device, log_writer=log_writer, args=args, prefix="NOCS")


    # Save more stuff
    write_log_stats(stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))



def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                            # shuffle = False,
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


@torch.no_grad()
def test(model, data_loader, device, args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
    #     data_loader.dataset.set_epoch(epoch)
    # if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
    #     data_loader.sampler.set_epoch(epoch)
    all_pt_errors = np.array([])
    all_rot_errors = np.array([])
    all_trans_errors = np.array([])
    all_class_error = {}

    for i, batch in enumerate(tqdm(data_loader)):
        ignore_keys = set(['depthmap', 'coords', 'focal_length', 'principal_point', 'xy_map', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
        for name in batch.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            # print(name)
            batch[name] = batch[name].to(device, non_blocking=True)

        pred1, pred2 = model(batch)
        pt_errors = pt_eval(pred1["pts3d"], batch["pts3d"], batch["valid_mask"])
        rot_errors, trans_errors, class_errors = pose_eval(pred2["pts3d"], batch["camera_pose"], batch["valid_mask"], batch["crop_params"], batch["focal_length"], batch["principal_point"], batch["class_id"], batch["mug_handle"])
        if i == 0:
            all_pt_errors = np.array(pt_errors)
            all_rot_errors = np.array(rot_errors)
            all_trans_errors = np.array(trans_errors)
            for key in class_errors:
                class_errors[key] = np.array(class_errors[key])
            all_class_errors = class_errors
        else:
            all_pt_errors = np.concatenate((all_pt_errors, np.array(pt_errors)), axis=0)
            all_rot_errors = np.concatenate((all_rot_errors, np.array(rot_errors)), axis=0)
            all_trans_errors = np.concatenate((all_trans_errors, np.array(trans_errors)), axis=0)
            for key in all_class_errors.keys():
                all_class_errors[key] = np.concatenate((all_class_errors[key], np.array(class_errors[key])), axis=0)
            # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    results = {
        "Mean Point Error": str(np.mean(all_pt_errors)),
        "Median Point Error": str(np.median(all_pt_errors)),
        "Mean Rotation Error": str(np.mean(all_rot_errors)),
        "Median Rotation Error": str(np.median(all_rot_errors)),
        "Mean Translation Error": str(np.mean(all_trans_errors)),
        "Median Translation Error": str(np.median(all_trans_errors)),
    }
    
    for key in all_class_errors.keys():
        results["Mean_" + key] = str(np.mean(all_class_errors[key]))
        results["Median_" + key] = str(np.median(all_class_errors[key]))
    

    print(results)

    return results
