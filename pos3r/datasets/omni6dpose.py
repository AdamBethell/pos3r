# --------------------------------------------------------
# Dataloader for preprocessed Omni6DPose
# dataset at https://github.com/Omni6DPose/Omni6DPoseAPI - MIT License
# --------------------------------------------------------
import os.path as osp
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

class Omni6DPose(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, mode="train", source="Omni6DPose", append_ndc=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'Omni6DPose'
        self.dynamic_zoom_in_params = dynamic_zoom_in_params
        self.deform_2d_params = deform_2d_params
        self.append_ndc = append_ndc
        self.mode = mode

        if mode == "real":
            d = osp.join(self.ROOT, '*')
        else:
            d = osp.join(self.ROOT, '*', 
                        ('' if mode == 'real' else mode), 
                        "color", ('' if source == 'Omni6DPose' else source), "*")

        img_list = Dataset.glob_prefix(root=d)
        assert len(img_list)
        self.img_list = img_list
        self.length = len(self.img_list)
        
        self.obj_meta = cutoop.obj_meta.ObjectMetaData.load_json(
            "../RGBGenPose/configs/obj_meta.json" if mode != 'real'
                else "../RGBGenPose/configs/real_obj_meta.json")
        self.cat_names = [cl.name for cl in self.obj_meta.class_list]
        self.cat_name2id = {name: i for i, name in enumerate(self.cat_names)}
        self.id2cat_name = {str(i): name for i, name in enumerate(self.cat_names)}

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

    def _get_views(self, idx):
        # choose a scene
        path = self.img_list[idx]
        # print(path)
        impath = path + "color.png"
        raw_rgb = Dataset.load_color(impath)
        im_H, im_W = raw_rgb.shape[0], raw_rgb.shape[1]
        # load camera params
        gts = Dataset.load_meta(path.replace("color", "meta") + "meta.json")
        if not any(obj.is_valid for obj in gts.objects):
                return self._get_views((idx + 1) % self.__len__())
        valid_objects = [obj for obj in gts.objects if obj.is_valid]
        obj = random.sample(valid_objects, 1)[0]
        # obj = valid_objects[1]
        inst_name = obj.meta.oid
        # print(inst_name)

        if inst_name not in self.obj_meta.instance_dict:
            return self._get_views((idx + 1) % self.__len__())
        
        maskmap = Dataset.load_mask(path.replace("color", "mask") + "mask.exr")
        object_mask = np.equal(maskmap, obj.mask_id)
        if not np.any(object_mask):
            return self._get_views((idx + 1) % self.__len__())
        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], im_H, im_W)
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        # print(bbox_xyxy)
        # bbox = jitter_bbox(
        #     bbox_xyxy,
        #     jitter_scale=(0.75, 1.25),
        #     jitter_trans=(-0.25,0.25),
        # )
        if self.mode == "train":
            bbox_centre, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)
            # bbox_centre, scale = aug_bbox_eval(bbox_xyxy, self.dynamic_zoom_in_params['DZI_PAD_SCALE'], im_H, im_W)
        else:
            bbox_centre, scale = aug_bbox_eval(bbox_xyxy, self.dynamic_zoom_in_params['DZI_PAD_SCALE'], im_H, im_W)

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
        rotation = quaternion_to_matrix(torch.tensor(obj.quaternion_wxyz))
        translation = torch.Tensor(obj.translation)
        camera_pose = torch.eye(4)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = translation

        depthmap_pose = torch.eye(4)
        depthmap_pose[:3, :3] = rotation.T
        depthmap_pose[:3, 3] = - rotation.T @ translation

        intrinsics = gts.camera.intrinsics
        img_resize_scale = raw_rgb.shape[0] / intrinsics.height
        assert raw_rgb.shape[1] / intrinsics.width == img_resize_scale
        mat_K = np.array([[intrinsics.fx, 0, intrinsics.cx, 0], [0, intrinsics.fy, intrinsics.cy, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                         dtype=np.float32)  # [fx, fy, cx, cy]
        mat_K *= img_resize_scale
        mat_K[2, 3] = 1
        mat_K[3, 2] = 1
        intrinsics = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]],
                         dtype=np.float32)
        intrinsics *= img_resize_scale

        l = min(im_W, im_H)
        f = mat_K[0,0]
        px, py = mat_K[:2, 2]
        focal_length_ndc = [2 * f / l, 2 * f / l]
        principal_point_ndc = [-2 * px / l + im_W / l, -2 * py / l + im_H / l]


        depthmap = Dataset.load_depth(path.replace("color", "depth_1") + "depth_1.exr")
        depthmap[depthmap > 3] = 0

        # load object mask
        mask_target = maskmap.copy().astype(np.float32)
        mask_target[maskmap != obj.mask_id] = 0.0
        mask_target[maskmap == obj.mask_id] = 1.0

        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        # rgb = self._crop_image(Image.fromarray(raw_rgb), bbox)
        # roi_mask = self._crop_image(Image.fromarray(mask_target), bbox)
        # roi_depth = self._crop_image(Image.fromarray(depthmap), bbox)
        # print(bbox_centre)
        # print(scale)
        rgb, trans = crop_resize_by_warp_affine(
                raw_rgb, bbox_centre, scale, self.resolution, interpolation=cv2.INTER_LINEAR
            )
        # print(trans)
        roi_mask, _ = crop_resize_by_warp_affine(
            mask_target, bbox_centre, scale, self.resolution, interpolation=cv2.INTER_NEAREST
        )

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

        roi_depth *= roi_mask

        # trans_inv = np.eye(3)
        # trans_inv[:2, :] = trans
        # trans_inv = np.linalg.inv(trans_inv)
        # # trans_inv[0,1] = 0
        # print(trans_inv)
        # intrinsics =  intrinsics @ trans_inv
        # intrinsics[0,1] = 0
        # intrinsics[1,0] = 0

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

        oid = obj.meta.oid
        objinfo = self.obj_meta.instance_dict[oid]
        size = objinfo.dimensions
        size = np.array(size).astype(np.float32)
        bbox_3d = obj.meta.bbox_side_len
        
        view = dict(
            img=rgb,
            depthmap=roi_depth,
            xy_map=roi_coord_2d,
            camera_pose=camera_pose,
            depthmap_pose=depthmap_pose,
            ray_pose=ray_pose,
            camera_intrinsics=intrinsics,
            focal_length=focal_length_ndc,
            principal_point=principal_point_ndc,
            mat_k=mat_K,
            crop_params=crop_params,
            # camera=camera,
            rays=rays[0, :-2, :, :].permute(1,2,0),
            coords=coords,
            dataset=self.dataset_label,
            label=path,
            instance=inst_name,
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
