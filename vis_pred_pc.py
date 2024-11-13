import open3d as o3d  
import numpy as np 
import torch
import math

from pos3r.model import RayCroCoNet
from pos3r.datasets.omni6dpose import Omni6DPose

DYNAMIC_ZOOM_IN_PARAMS = {
    'DZI_PAD_SCALE': 1.5,
    'DZI_TYPE': 'uniform',
    'DZI_SCALE_RATIO': 0.25,
    'DZI_SHIFT_RATIO': 0.25
}
# pts aug parameters
PTS_AUG_PARAMS = {
    'aug_pc_pro': 0.2,
    'aug_pc_r': 0.2,
    'aug_rt_pro': 0.3,
    'aug_bb_pro': 0.3,
    'aug_bc_pro': 0.3
}
# 2D aug parameters
DEFORM_2D_PARAMS = {
    'roi_mask_r': 3,
    'roi_mask_pro': 0.5
    }

def main():
    dataset = Omni6DPose(ROOT="../RGBGenPose/data/SOPE_Test", resolution=224)
    pretrained = "checkpoints/pos3r/checkpoint-best.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    batch = dataset.__getitem__(0)

    print('Loading model')
    model = RayCroCoNet(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -math.inf, math.inf), conf_mode=None, enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print('Loading pretrained: ', pretrained)
    ckpt = torch.load(pretrained, map_location=device)
    print(model.load_state_dict(ckpt['model'], strict=False))
    del ckpt  # in case it occupies memory

    keys = set(['pts3d', 'valid_mask', 'img'])
    for name in batch.keys():  # pseudo_focal
        if name in keys:
            # print(name)
            batch[name] = torch.Tensor(batch[name]).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        pred1, pred2 = model(batch)

    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud()
    print(batch["instance"])
    pts = np.concatenate((batch["pts3d"][batch["valid_mask"].cpu().numpy() > 0].cpu().numpy(), pred1["pts3d"][batch["valid_mask"].cpu().numpy() > 0].cpu().numpy()), axis=0)
    pcd.point.positions = o3d.core.Tensor(pts.tolist(), dtype)
    pcd.point.colors = o3d.core.Tensor(np.concatenate((np.zeros_like(batch["pts3d"][batch["valid_mask"].cpu().numpy() > 0].cpu().numpy()),np.ones_like(batch["pts3d"][batch["valid_mask"].cpu().numpy() > 0].cpu().numpy())), axis=0) , dtype)
    print(pcd)
    o3d.visualization.draw([pcd])    # Visualize point cloud   
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry([pcd])
    # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    # vis.run()   

if __name__ == "__main__":
    main()