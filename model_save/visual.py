import os
import math
import numpy as np

import cv2
import skimage.io as io
import open3d as open3d
from skimage.color import rgb2gray
import torch
from model_utility import Visualization



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Scene Flow 비주얼라이지션을 위한 모듈 모음
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TAG_CHAR             = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH  = 1e7

sampling             = [4,20,25,35,40]
imgflag              = 1 # 0 is image, 1 is flow

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0912

cam_center_dict = dict()
cam_center_dict[1242] = [6.095593e+02, 1.728540e+02]
cam_center_dict[1241] = [6.071928e+02, 1.852157e+02]
cam_center_dict[1224] = [6.040814e+02, 1.805066e+02]
cam_center_dict[1238] = [6.003891e+02, 1.815122e+02]
cam_center_dict[1226] = [6.018873e+02, 1.831104e+02]


def numpy2torch(array):
    assert(isinstance(array, np.ndarray))
    """
    넘파이 어레이가 [H, W, C]이면 array.ndim == 3이고
        (0, 1, 2) -> (2, 0, 1)으로 transpose하면 [C, H, W] 형태로 바뀜
        ndim == 3이 아니면 disparity이고 [H, W]이고 axis = 0을 늘려서 [1, H, W]로 변환
    """
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array.copy()).float()



def vis_create_pixelgrid(b, h, w, flow=None):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)    
    ones   = torch.ones_like(grid_h)

    if flow is None:
        pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False)
    else:
        pixelgrid = torch.cat((grid_h + flow[:, 0:1, :, :], grid_v + flow[:, 1:2, :, :], ones), dim=1)
        pixelgrid = pixelgrid.float().requires_grad_(False)
    return pixelgrid


def vis_disp2depth_kitti(pred_disp, focal):
    pred_depth = focal * 0.54 / pred_disp
    pred_depth = torch.clamp(pred_depth, 1e-3, 80)
    return pred_depth


def vis_pixel2point(depth, intrinsic, flow=None):
    b, _, h, w = depth.size()    
    pixelgrid  = vis_create_pixelgrid(b, h, w, flow)
    depth_mat  = depth.view(b, 1, -1)    
    pixel_mat  = pixelgrid.view(b, 3, -1)

    pts_mat    = torch.matmul(torch.inverse(intrinsic), pixel_mat) * depth_mat
    pts        = pts_mat.view(b, -1, h, w)
    return pts, pixelgrid


def vis_pixel2point_ms(output_disp, intrinsic, flow=None):
    focal = intrinsic[:, 0, 0]
    output_depth = vis_disp2depth_kitti(output_disp, focal)
    pts, _       = vis_pixel2point(output_depth, intrinsic, flow)
    return pts


def create_point_cloud(iml1_path, flow_path, disp1_path, disp2_path, tt = 0):
    iml1_np  = io.imread(iml1_path) / np.float32(255.0)
    flow_np  = Visualization().read_flow(flow_path)
    disp1_np = Visualization().read_disp(disp1_path)
    disp2_np = Visualization().read_disp(disp2_path)
    
    ## numpy2torch에서 [C H W] or [1 H W]으로 바꾸고 unsqueeze(0)을 통해 [1 C H W] or [1 1 H W]으로 바꿈
    im1   = numpy2torch(iml1_np).unsqueeze(0)
    flow  = numpy2torch(flow_np).unsqueeze(0)
    disp1 = numpy2torch(disp1_np).unsqueeze(0)
    disp2 = numpy2torch(disp2_np).unsqueeze(0)


    ## B C H W
    B, C, H, W = im1.size()
    ## Intrinsic
    focal = width_to_focal[W]
    cx    = cam_center_dict[W][0]
    cy    = cam_center_dict[W][1]
    kl    = numpy2torch(np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]]))

    ## Forward warping Pts1 using disp_change and flow
    ## torch.Size([1, 3, 370, 1224])
    ## disp1을 depth1로 바꾸고 point cloud 와핑
    pts1      = vis_pixel2point_ms(disp1, kl)
    ## disp2를 depth2로 바꾸고 flow를 더한 좌표를 kl로 projection
    pts1_warp = vis_pixel2point_ms(disp2, kl, flow)
    scene_f   = pts1_warp - pts1

    ## Composing Image
    im1_np0_g = np.repeat(np.expand_dims(rgb2gray(iml1_np), axis=2), 3, axis=2)
    flow      = torch.cat((scene_f[:, 0:1, :, :], scene_f[:, 2:3, :, :]), dim=1)
    flow      = flow.data.cpu().numpy()[0, :, :, :]
    flow_img  = Visualization().flow2png(flow) / np.float32(255.0)
    
    if imgflag == 0:
        flow_img = iml1_np
    else:
        flow_img = (flow_img * 0.75 + im1_np0_g * 0.25)
    
    ## Crop
    max_crop = (60, 0.7, 82)
    min_crop = (-60, -20, 0)

    x1 = -60
    x2 = 60
    y1 = 0.7
    y2 = -20
    z1 = 80
    z2 = 0
    pp1 = np.array([[x1, y1, z1]])
    pp2 = np.array([[x1, y1, z2]])
    pp3 = np.array([[x1, y2, z1]])
    pp4 = np.array([[x1, y2, z2]])
    pp5 = np.array([[x2, y1, z1]])
    pp6 = np.array([[x2, y1, z2]])
    pp7 = np.array([[x2, y2, z1]])
    pp8 = np.array([[x2, y2, z2]])
    bb_pts = np.concatenate((pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8), axis=0)
    wp = np.array([[1.0, 1.0, 1.0]])
    bb_colors = np.concatenate((wp, wp, wp, wp, wp, wp, wp, wp), axis=0)

    ## Open3D Vis
    pts1_tform = pts1 + scene_f*tt
    pts1_np    = np.transpose(pts1_tform[0].view(3, -1).data.numpy(), (1, 0))
    pts1_np    = np.concatenate((pts1_np, bb_pts), axis=0)
    pts1_color = np.reshape(flow_img, (H * W, 3))
    pts1_color = np.concatenate((pts1_color, bb_colors), axis=0)

    pcd1        = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(pts1_np)
    pcd1.colors = open3d.utility.Vector3dVector(pts1_color)

    bbox = open3d.geometry.AxisAlignedBoundingBox(min_crop, max_crop)
    pcd1 = pcd1.crop(bbox)
    return pcd1


def custom_vis(iml1_path, flow_path, disp1_path, disp2_path, save_name):
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    custom_vis.index = 0
    custom_vis.trajectory = open3d.io.read_pinhole_camera_trajectory("cam_pose.json")
    custom_vis.vis        = open3d.visualization.Visualizer()
    init_pcd              = create_point_cloud(iml1_path, flow_path, disp1_path, disp2_path, 0)
    custom_vis.prev_pcd   = init_pcd

    def move_forward(vis):
        glb   = custom_vis
        ## Capture
        depth = vis.capture_depth_float_buffer(False)
        image = vis.capture_screen_float_buffer(False)
        file_name  = ""
        image_name = iml1_path.split("/")[-1]

        if imgflag == 0:
            file_name = os.path.join(save_name, "{}_{}".format(image_name[8:17], glb.index) + ".png")
        else:
            file_name = os.path.join(save_name, "{}_{}".format(image_name[8:17], glb.index) + ".png")
        print(file_name)
        io.imsave(file_name, np.asarray(image), check_contrast=False)

        ## Rendering
        max_d_x = 13
        max_d_y = 4
        
        if glb.index < sampling[0]:
            tt = 0
            rx = 0
            ry = 0
        elif glb.index < sampling[1]: # only rotation
            tt = 0 
            rad = 2 * 3.14159265359 / (sampling[1] - sampling[0]) * (glb.index - sampling[0])
            rx = max_d_x * math.sin(rad)
            ry = (max_d_y * math.cos(rad) - max_d_y)
        elif glb.index < sampling[2]:
            tt = 0
            rx = 0
            ry = 0
        elif glb.index < sampling[3]:
            tt = (glb.index - sampling[2]) / (sampling[3] - sampling[2]) 
            rx = 0
            ry = 0
        else:
            tt = 1
            rx = 0
            ry = 0

        # img_id = imglist[glb.index]
        pcd = create_point_cloud(iml1_path, flow_path, disp1_path, disp2_path, 0)
        glb.index = glb.index + 1

        vis.clear_geometries()
        vis.add_geometry(pcd)
        glb.prev_pcd = pcd

        ctr = vis.get_view_control()
        ctr.scale(-24)

        ctr.rotate(rx, 980.0  + ry, 0, 0)
        ctr.translate(-5, 0)
        if glb.index == 21:
            custom_vis.vis.register_animation_callback(None)
        return False

    vis = custom_vis.vis
    vis.create_window()
    vis.add_geometry(init_pcd)

    ctr = vis.get_view_control()
    ctr.scale(-24)
    ctr.rotate(0, 980.0, 0, 0)
    ctr.translate(-5, 0)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    iml1_path_list  = []
    flow_path_list  = []
    disp1_path_list = []
    disp2_path_list = []
    for index in range(0, 200):
        iml1_path  = os.path.join("./image_2", "%.6d_10" % index + ".jpg")

        flow_path  = os.path.join("./custom_flow", "%.6d_10" % index + ".png")
        disp1_path = os.path.join("./custom_disp1", "%.6d_10" % index + ".png")
        disp2_path = os.path.join("./custom_disp2", "%.6d_11" % index + ".png")
        # flow_path  = os.path.join("./official_flow", "%.6d_10" % index + ".png")
        # disp1_path = os.path.join("./official_disp1", "%.6d_10" % index + ".png")
        # disp2_path = os.path.join("./official_disp2", "%.6d_11" % index + ".png")

        iml1_path_list.append(iml1_path)
        flow_path_list.append(flow_path)
        disp1_path_list.append(disp1_path)
        disp2_path_list.append(disp2_path)

    sample_index = [0, 10, 23]
    for index in sample_index:
        custom_vis(iml1_path_list[index], flow_path_list[index], disp1_path_list[index], disp2_path_list[index], "custom")