# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from pos3r.utils.device import to_cpu, collate_with_cat
from pos3r.utils.misc import invalid_to_nans
from pos3r.utils.geometry import depthmap_to_pts3d, geotrf
from pos3r.eval_utils import pt_eval, pose_eval_nocs, pose_eval_omni


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(batch, model, criterion, device, use_amp=False, ret=None, i=0, test=False):
    ignore_keys = set(['depthmap', 'coords', 'focal_length', 'principal_point', 'xy_map', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', "class_id", "sym_label", "mug_handle"])
    for name in batch.keys():  # pseudo_focal
        if name in ignore_keys:
            continue
        # print(name)
        batch[name] = batch[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(batch)
        if i % 100 == 0:
            print(pred1['pts3d'][batch["valid_mask"]][0])
            print(pred1['pts3d'][batch["valid_mask"]][-1])
            print(batch["pts3d"][batch["valid_mask"]][0])
            print(batch["pts3d"][batch["valid_mask"]][-1])
            # print(pred1["conf"][:, 112, 112])
            print(pred2['pts3d'][batch["valid_mask"]][0])
            print(pred2["pts3d"][batch["valid_mask"]][-1])
            print(batch['rays'][batch["valid_mask"]][0])
            print(batch["rays"][batch["valid_mask"]][-1])
            # print(pred2["conf"][:, 112, 112])
        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(batch["pts3d"], batch["rays"], pred1, pred2, batch["valid_mask"]) if criterion is not None else None
        if test:
            pt_errors = pt_eval(pred1["pts3d"], batch["pts3d"], batch["valid_mask"])
            if batch["dataset"][0] == "NOCS":
                rot_errors, trans_errors, _ = pose_eval_nocs(pred2["pts3d"], batch["camera_pose"], batch["valid_mask"], batch["crop_params"], batch["focal_length"], batch["principal_point"], batch["class_id"], batch["mug_handle"], class_res=False)
            elif batch["dataset"][0] == "Omni6DPose":
                rot_errors, trans_errors, _ = pose_eval_omni(pred2["pts3d"], batch["camera_pose"], batch["valid_mask"], batch["crop_params"], batch["focal_length"], batch["principal_point"], batch["class_id"], batch["sym_label"], class_res=False)
            else:
                print("Invalid Dataset")
                return {}
            loss_value, loss_details = loss
            loss_details["Point_Error"] = torch.mean(torch.Tensor(pt_errors))
            loss_details["Rotation_Error"] = torch.mean(torch.Tensor(rot_errors))
            loss_details["Translation_Error"] = torch.mean(torch.Tensor(trans_errors))
            loss = loss_value, loss_details
        
    result = dict(batch=batch, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
