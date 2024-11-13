import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
import cutoop
from cutoop.rotation import rot_canonical_sym, SymLabel
from pos3r.rays import masked_rays_to_cameras, Rays

synset_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
torch3d_T_colmap = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
objmeta = cutoop.obj_meta.ObjectMetaData.load_json("../RGBGenPose/data/META/obj_meta.json")
class_names = [cl.name for cl in objmeta.class_list]

def array_to_SymLabel(arr_Nx4: np.ndarray):
    syms_N = []
    tags = ['none', 'any', 'half', 'quarter']
    for a, x, y, z in arr_Nx4:
        syms_N.append(SymLabel(bool(a), tags[x], tags[y], tags[z]))
    return syms_N

def calculate_rot_error_nocs(pred_rot, gt_rot, class_id, handle_visibility):
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

def calculate_rot_error_omni(pred_rot, gt_rot, sym_label):
    pred_rot = torch3d_T_colmap @ pred_rot.T
    s1 = np.cbrt(np.linalg.det(pred_rot.cpu().numpy()))
    R1 = pred_rot.cpu().numpy() / s1
    s2 = np.cbrt(np.linalg.det(gt_rot.cpu().numpy()))
    R2 = gt_rot.cpu().numpy() / s2

    sym_label = array_to_SymLabel(sym_label.unsqueeze(0).numpy())
    if str(sym_label[0]) == "any":
        R_error = 0
    else:
        rot, R_error = rot_canonical_sym(R2, R1, sym_label[0], return_theta=True)
    
    return R_error * 180 / np.pi

def calculate_trans_error(pred_trans, gt_trans):
    pred_trans = torch3d_T_colmap @ pred_trans
    trans_error = np.linalg.norm(pred_trans.cpu().numpy() - gt_trans.cpu().numpy())
    return trans_error

def pt_eval(pred, gt, mask):
    pred_masked = (pred * mask.unsqueeze(-1)).cpu().numpy()
    gt_masked = (gt * mask.unsqueeze(-1)).cpu().numpy()
    
    dist_error =  np.linalg.norm(pred_masked - gt_masked, axis=0)
    
    return dist_error


def pose_eval_nocs(rays_final, gt, mask, crop_parameters, focal_length, principal_point, class_ids, mug_handles, class_res=True):
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point)
    
    rays = Rays.from_spatial(rays_final.permute(0, 3, 1, 2).cpu())
    pred_cameras = masked_rays_to_cameras(rays, crop_parameters.cpu(), mask.reshape(mask.shape[0], -1).cpu(), cameras=cameras) 
    
    rot_errors = []
    trans_errors = []
    rot_class_errors = {}
    trans_class_errors = {}
    if class_res:
        for name in synset_names:
            rot_class_errors[name] = []
            trans_class_errors[name] = []
    for i in range(gt.shape[0]):
        rot_error = calculate_rot_error_nocs(pred_cameras.R[i], gt[i, :3, :3], class_ids[i], mug_handles[i])
        trans_error = calculate_trans_error(pred_cameras.T[i], gt[i, :3, 3])
        rot_errors.append(rot_error)
        trans_errors.append(trans_error)
        if class_res:
            rot_class_errors[class_names[class_ids[i]]].append(rot_error)
            trans_class_errors[class_names[class_ids[i]]].append(trans_error)
    return rot_errors, trans_errors, rot_class_errors, trans_class_errors


def pose_eval_omni(rays_final, gt, mask, crop_parameters, focal_length, principal_point, class_ids, sym_labels, class_res=True):
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point)
    
    rays = Rays.from_spatial(rays_final.permute(0, 3, 1, 2).cpu())
    pred_cameras = masked_rays_to_cameras(rays, crop_parameters.cpu(), mask.reshape(mask.shape[0], -1).cpu(), cameras=cameras) 
    
    rot_errors = []
    trans_errors = []
    rot_class_errors = {}
    trans_class_errors = {}
    if class_res:
        for name in class_names:
            rot_class_errors[name] = []
            trans_class_errors[name] = []
    for i in range(gt.shape[0]):
        rot_error = calculate_rot_error_omni(pred_cameras.R[i], gt[i, :3, :3], sym_labels[i])
        trans_error = calculate_trans_error(pred_cameras.T[i], gt[i, :3, 3])
        rot_errors.append(rot_error)
        trans_errors.append(trans_error)
        if class_res:
            rot_class_errors[class_names[class_ids[i]]].append(rot_error)
            trans_class_errors[class_names[class_ids[i]]].append(trans_error)
    
    return rot_errors, trans_errors, rot_class_errors, trans_class_errors
    

def compute_RT_matches(rot_errors, trans_errors, degree_thres_list, shift_thres_list):
    matches = np.zeros((rot_errors.shape[0], len(degree_thres_list), len(shift_thres_list)))
    for i in rot_errors.shape[0]:
        for d in degree_thres_list:
            for s in degree_thres_list:
                if rot_errors[i] <= d and trans_errors[i] <= s:
                    matches[i, d, s] = 1
    
    return matches

def com_acc_ap(matches):
    if len(matches) > 0:
        acc = sum(matches) / len(matches)
    else:
        acc = 0

    precisions = np.cumsum(matches) / (np.arange(len(matches)) + 1)
    recalls = np.cumsum(matches).astype(np.float32) / len(matches)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    # compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return acc, ap

def compute_accs_and_aps(all_matches, class_matches):
    _, ds, ss = all_matches.shape

    all_acc = np.zeros((ds, ss))
    all_aps = np.zeros((ds, ss))
    class_acc = {}
    class_aps = {}

    for d in range(ds):
        for s in range(ss):
            all_acc[d,s], all_aps[d,s] = com_acc_ap(all_matches[:, d, s])

    for key in class_matches.keys():
        class_acc[key] = np.zeros((ds, ss))
        class_aps[key] = np.zeros((ds, ss))
        for d in range(ds):
            for s in range(ss):
                class_acc[key][d,s], class_aps[key][d,s] = com_acc_ap(class_matches[key][:, d, s])
    
    return all_acc, all_aps, class_acc, class_aps

