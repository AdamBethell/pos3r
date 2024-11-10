import open3d as o3d  
import numpy as np 
import torch
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
    view1 = dataset.__getitem__(0)
    while "carrot" not in view1["instance"]:
        view1 = dataset.__getitem__(0)
    view2 = dataset.__getitem__(0)
    # while view1["instance"] == view2["instance"]:
    #     view2 = dataset.__getitem__(0)
    while "bottle" not in view2["instance"]:
        view2 = dataset.__getitem__(0)
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud()
    print(view1["instance"])
    print(view2["instance"])
    pts = np.concatenate((view1["pts3d"][view1["valid_mask"]], view2["pts3d"][view2["valid_mask"]]), axis=0)
    pcd.point.positions = o3d.core.Tensor(pts.tolist(), dtype)
    pcd.point.colors = o3d.core.Tensor(np.zeros_like(pts), dtype)
    print(pcd)
    o3d.visualization.draw([pcd])    # Visualize point cloud   
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry([pcd])
    # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    # vis.run()   

if __name__ == "__main__":
    main()