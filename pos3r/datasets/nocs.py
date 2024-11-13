# --------------------------------------------------------
# Dataloader for preprocessed Omni6DPose
# dataset at https://github.com/Omni6DPose/Omni6DPoseAPI - MIT License
# --------------------------------------------------------
import os.path as osp
import os
import json
import itertools
import random
import cutoop
import torch
from cutoop.data_loader import Dataset
from cutoop.rotation import SymLabel
from cutoop.eval_utils import *
from cutoop.transform import *
from pytorch3d.renderer import PerspectiveCameras
from pos3r.rays import cameras_to_rays
from torchvision import transforms
from PIL import Image
import _pickle as cPickle


import cv2
import numpy as np

from pos3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from pos3r.datasets.datasets_utils import aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine, aug_bbox_eval, defor_2D

dynamic_zoom_in_params = {
    'DZI_PAD_SCALE': 1.5,
    'DZI_TYPE': 'uniform',
    'DZI_SCALE_RATIO': 0.25,
    'DZI_SHIFT_RATIO': 0.25
}

deform_2d_params = {
    'roi_mask_r': 3,
    'roi_mask_pro': 0.5
    }


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_bbox(bbox, img_width=480, img_length=640):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def jitter_bbox(square_bbox, jitter_scale=(1.1, 1.2), jitter_trans=(-0.07, 0.07)):
    square_bbox = np.array(square_bbox.astype(float))
    s = np.random.uniform(jitter_scale[0], jitter_scale[1])
    tx, ty = np.random.uniform(jitter_trans[0], jitter_trans[1], size=2)
    side_length = square_bbox[2] - square_bbox[0]
    center = (square_bbox[:2] + square_bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s
    ul = center - extent
    lr = ul + 2 * extent
    return np.concatenate((ul, lr))

def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(0, width-1, width, dtype=np.float32)
    y = np.linspace(0, height-1, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy

def load_depth(img_path):
    """ Load depth image from img_path. """
    if img_path[-4:] == '.png':
        depth_path = img_path
    else:
        depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    depth16 = depth16.astype(np.float32) / 1000
    return depth16

class NOCS(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, mode="train", source="CAMERA+Real", append_ndc=True, all_objects=False, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'NOCS'
        self.dynamic_zoom_in_params = dynamic_zoom_in_params
        self.deform_2d_params = deform_2d_params
        self.append_ndc = append_ndc
        self.mode = mode
        self.source = source
        self.all_objects = all_objects


        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        if mode == 'train':
            del img_list_path[2:]
        else:
            del img_list_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
        elif source == 'Real':
            del img_list_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(self.ROOT, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        if source == 'CAMERA':
            self.id2cat_name = self.id2cat_name_CAMERA
        assert len(img_list)
        
        if self.all_objects:
            new_img_list = []
            self.obj_idxs = []
            for img in img_list:
                with open(os.path.join(self.ROOT, img  + '_label.pkl'), 'rb') as f:
                    gts = cPickle.load(f)
                for i in range(len(gts["instance_ids"])):
                    new_img_list.append(img)
                    self.obj_idxs.append(i)
            self.img_list = new_img_list
        else:
            self.img_list = img_list
        self.length = len(self.img_list)

        with open(os.path.join(self.ROOT, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float32)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)

        self.invaild_list = []
        with open(os.path.join(self.ROOT, 'Real/train/mug_handle.pkl'), 'rb') as f:
            self.mug_sym = cPickle.load(f)

    def __len__(self):
        return self.length


    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def _get_views(self, index):
        # choose a scene
        img_path = os.path.join(self.ROOT, self.img_list[index])
        if img_path in self.invaild_list:
            return self.__getitem__((index + 1) % self.__len__())
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())

        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'
        if self.all_objects:
            idx = self.obj_idxs[index]
        else:
            idx = random.randint(0, len(gts['instance_ids']) - 1)
        inst_name = self.id2cat_name[str(gts['class_ids'][idx])]
        if gts['class_ids'][idx] == 6 and img_type == 'real':
            if self.mode == 'train':
                handle_tmp_path = img_path.split('/')
                scene_label = handle_tmp_path[-2] + '_res'
                img_id = int(handle_tmp_path[-1])
                mug_handle = self.mug_sym[scene_label][img_id]
            else:
                mug_handle = gts['handle_visibility'][idx]
        else:
            mug_handle = 1

        rgb = cv2.imread(img_path + '_color.png')
        mask = cv2.imread(img_path + '_mask.png')
        depth_path = img_path + '_depth.png'
        
        if os.path.exists(depth_path):
            depthmap = load_depth(depth_path)
        else:
            return self.__getitem__((index + 1) % self.__len__())
        mask = mask[:, :, 2]
        im_H, im_W = rgb.shape[0], rgb.shape[1]
        
        # aggragate information about the selected object
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bw = (bbox_xyxy[2] - bbox_xyxy[0])
        bh = (bbox_xyxy[3] - bbox_xyxy[1])
        if self.mode == "train":
            bbox_centre, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)
            # bbox_centre, scale = aug_bbox_eval(bbox_xyxy, self.dynamic_zoom_in_params['DZI_PAD_SCALE'], im_H, im_W)
        else:
            bbox_centre, scale = aug_bbox_eval(bbox_xyxy, self.dynamic_zoom_in_params['DZI_PAD_SCALE'], im_H, im_W)

        inst_id = gts['instance_ids'][idx]
        mask_target = mask.copy().astype(np.float32)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0

        rgb, trans = crop_resize_by_warp_affine(
                rgb, bbox_centre, scale, self.resolution, interpolation=cv2.INTER_LINEAR
            )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        roi_mask, _ = crop_resize_by_warp_affine(
            mask_target, bbox_centre, scale, self.resolution, interpolation=cv2.INTER_NEAREST
        )
        # bbox = np.around(bbox).astype(int)
        
        # # Crop parameters
        # crop_center = (bbox[:2] + bbox[2:]) / 2
        # # convert crop center to correspond to a "square" image
        # width, height = im_W, im_H
        # length = max(width, height)
        # s = length / min(width, height)
        # crop_center = crop_center + (length - np.array([width, height])) / 2
        # # convert to NDC
        # cc = s - 2 * s * crop_center / length
        # crop_width = 2 * s * (bbox[2] - bbox[0]) / length
        # crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])
        
        length = max(im_H, im_W)
        s = length / min(im_W, im_H)
        # bbox_centre = bbox_centre + (length - np.array([im_W, im_H])) / 2
        # print(bbox_centre)
        # print(scale)
        crop_center = bbox_centre + (length - np.array([im_W, im_H])) / 2
        cc = s - 2 * s * crop_center / length
        crop_width = 2 * s * scale / length
        crop_params = torch.tensor([-cc[0], -cc[1], crop_width, scale])

        # print(crop_params)
        
        class_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        # note that this is nocs model, normalized along diagonal axis
        model_name = gts['model_list'][idx]
        rotation = torch.Tensor(gts['rotations'][idx]).to(torch.float32)
        translation = torch.Tensor(gts['translations'][idx])
        # size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32)
        
        camera_pose = torch.eye(4)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = translation

        depthmap_pose = torch.eye(4)
        depthmap_pose[:3, :3] = rotation.T
        depthmap_pose[:3, 3] = - rotation.T @ translation

        l = min(im_W, im_H)
        f = out_camK[0,0]
        px, py = out_camK[:2, 2]
        focal_length_ndc = [2 * f / l, 2 * f / l]
        principal_point_ndc = [-2 * px / l + im_W / l, -2 * py / l + im_H / l]

        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        # rgb = self._crop_image(Image.fromarray(raw_rgb), bbox)
        # roi_mask = self._crop_image(Image.fromarray(mask_target), bbox)
        # roi_depth = self._crop_image(Image.fromarray(depthmap), bbox)
        # print(bbox_centre)
        # print(scale)

        roi_depth, _ = crop_resize_by_warp_affine(
            depthmap, bbox_centre, scale, self.resolution, interpolation=cv2.INTER_NEAREST
        )

        roi_coord_2d, _ = crop_resize_by_warp_affine(
            coord_2d, bbox_centre, scale, self.resolution, interpolation=cv2.INTER_NEAREST
        )

        roi_coord_2d = roi_coord_2d.transpose(2, 0, 1)

        # roi_mask_def = defor_2D(
        #     roi_mask, 
        #     rand_r=self.deform_2d_params['roi_mask_r'], 
        #     rand_pro=self.deform_2d_params['roi_mask_pro']
        # )

        roi_depth *= roi_mask.astype(np.bool_)

        # trans_inv = np.eye(3)
        # trans_inv[:2, :] = trans
        # trans_inv = np.linalg.inv(trans_inv)
        # # trans_inv[0,1] = 0
        # print(trans_inv)
        # intrinsics =  intrinsics @ trans_inv
        # intrinsics[0,1] = 0
        # intrinsics[1,0] = 0
        intrinsics = out_camK

        num_valid = (roi_depth > 0.0).sum()
        if num_valid == 0:
           return self._get_views((idx + 1) % self.__len__())
        torch3d_T_colmap = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
        ray_R = (torch3d_T_colmap @ camera_pose[:3, :3]).T
        ray_T = torch3d_T_colmap @ camera_pose[:3, 3]
        ray_pose = torch.eye(4)
        ray_pose[:3, :3] = ray_R
        ray_pose[:3, 3] = ray_T
        camera = PerspectiveCameras(focal_length=torch.Tensor([focal_length_ndc]), principal_point=torch.Tensor([principal_point_ndc]), R=ray_R.unsqueeze(0), T=ray_T.unsqueeze(0))

        r = cameras_to_rays(
            cameras=camera,
            num_patches_x=self.resolution,
            num_patches_y=self.resolution,
            crop_parameters=crop_params.unsqueeze(0),
        )
        rays = r.to_spatial(include_ndc_coordinates=self.append_ndc)
        coords = rays[0, -2:, :, :]
        coords = torch.cat((coords, torch.ones(1, self.resolution, self.resolution)), dim=0)

        # oid = obj.meta.oid
        # objinfo = self.obj_meta.instance_dict[oid]
        # size = objinfo.dimensions
        # size = np.array(size).astype(np.float32)
        # bbox_3d = obj.meta.bbox_side_len

        view = dict(
            img=rgb,
            depthmap=roi_depth,
            xy_map=roi_coord_2d,
            camera_pose=camera_pose,
            depthmap_pose=depthmap_pose,
            ray_pose=ray_pose,
            camera_intrinsics=intrinsics,
            focal_length=torch.Tensor(focal_length_ndc),
            principal_point=torch.Tensor(principal_point_ndc),
            mat_k=out_camK,
            crop_params=crop_params,
            # camera=camera,
            rays=rays[0, :-2, :, :].permute(1,2,0),
            coords=coords,
            dataset=self.dataset_label,
            label=img_path,
            instance=inst_name,
            mug_handle=mug_handle,
            class_id=class_id
            # size=size,
            # bbox_3d=bbox_3d
        )
        return view


if __name__ == "__main__":
    from pos3r.datasets.base.base_stereo_view_dataset import view_name
    from pos3r.viz import SceneViz, auto_cam_size
    from pos3r.utils.image import rgb

    dataset = Omni6DPose(split='train', ROOT="../RGBGenPose/data/SOPE", resolution=224)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
