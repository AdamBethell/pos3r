import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras

from pos3r.rays import masked_rays_to_cameras, Rays

synset_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
torch3d_T_colmap = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()

def calculate_rot_error(pred_rot, gt_rot, class_id, handle_visibility):
    pred_rot = torch3d_T_colmap @ pred_rot.T
    s1 = np.cbrt(np.linalg.det(pred_rot.cpu().numpy()))
    R1 = pred_rot.cpu().numpy() / s1
    s2 = np.cbrt(np.linalg.det(gt_rot.cpu().numpy()))
    R2 = gt_rot.cpu().numpy() / s2
    
    if synset_names[class_id] in ['bottle', 'can', 'bowl', 'glass'] or \
        (synset_names[class_id] == 'mug' and handle_visibility == 0):
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    else:
        R = R1 @ R2.transpose()
        cos_theta = (np.trace(R) - 1) / 2
    R_error = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    
    return R_error

def calculate_trans_error(pred_trans, gt_trans):
    pred_trans = torch3d_T_colmap @ pred_trans
    trans_error = np.linalg.norm(pred_trans.cpu().numpy() - gt_trans.cpu().numpy())
    return trans_error

def pt_eval(pred, gt, mask):
    pred_masked = (pred * mask.unsqueeze(-1)).cpu().numpy()
    gt_masked = (gt * mask.unsqueeze(-1)).cpu().numpy()
    
    dist_error =  np.linalg.norm(pred_masked - gt_masked, axis=0)
    
    return dist_error


def pose_eval(rays_final, gt, mask, crop_parameters, focal_length, principal_point, class_ids, mug_handles):
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point)
    
    rays = Rays.from_spatial(rays_final.permute(0, 3, 1, 2).cpu())
    pred_cameras = masked_rays_to_cameras(rays, crop_parameters.cpu(), mask.reshape(mask.shape[0], -1).cpu(), cameras=cameras) 
    
    rot_errors = []
    trans_errors = []
    class_errors = {}
    for name in synset_names:
        class_errors[name + "_rotation_error"] = []
        class_errors[name + "_translation_error"] = []
    for i in range(gt.shape[0]):
        rot_error = calculate_rot_error(pred_cameras.R[i], gt[i, :3, :3], class_ids[i], mug_handles[i])
        trans_error = calculate_trans_error(pred_cameras.T[i], gt[i, :3, 3])
        rot_errors.append(rot_error)
        trans_errors.append(trans_error)
        class_errors[synset_names[class_ids[i]] + "_rotation_error"].append(rot_error)
        class_errors[synset_names[class_ids[i]] + "_translation_error"].append(trans_error)
    
    return rot_errors, trans_errors, class_errors
    