from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



####################################################################################################
######## 이미지 1차 미분, 2차 미분 계산 함수
######## 두 이미지 또는 플로우 간의 차이를 계산
######## 순서대로 L2 norm, L1 norm, robust norm
######## tensor concat, upsample, interpolation, grid_sample etc 계산 함수 모음
####################################################################################################
def gradient_x(img):
    img = F.pad(img, (0, 1, 0, 0), mode = "replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


def gradient_y(img):
    img = F.pad(img, (0, 0, 0, 1), mode = "replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


def gradient_x_2nd(img):
    img_l = F.pad(img, (1, 0, 0, 0), mode = "replicate")[:, :, :, :-1]
    img_r = F.pad(img, (0, 1, 0, 0), mode = "replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx


def gradient_y_2nd(img):
    img_t = F.pad(img, (0, 0, 1, 0), mode = "replicate")[:, :, :-1, :]
    img_b = F.pad(img, (0, 0, 0, 1), mode = "replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy


def elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p = 1, dim = 1, keepdim = True)


def elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p = 2 , dim = 1, keepdim = True)


def elementwise_robust_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    norm     = torch.norm(residual, p = 2, dim = 1, keepdim = True)
    return torch.pow(norm + 0.01, 0.4)


def bchw2bhwc(inputs):
    return inputs.transpose(1,2).transpose(2,3)


def bhwc2bchw(inputs):
    return inputs.transpose(2,3).transpose(1,2)


def concat(inputs_list, dim):
    return torch.cat(inputs_list, dim = dim)


def upsample(inputs, mode = "nearest", align_corners = False):
    return F.interpolate(inputs, scale_factor = 2, mode = mode, align_corners = align_corners)


def interpolate(inputs, size, mode = "bilinear", align_corners = True):
    return F.interpolate(inputs, size, mode = mode, align_corners = align_corners)


def interpolate_as(inputs, target_as, mode = "bilinear", align_corners = True):
    _, _, H, W = target_as.size()
    return F.interpolate(inputs, [H, W], mode = mode, align_corners = align_corners)


def grid_sample(inputs, coords, padding_mode = "border", align_corners = True):
    """
    padding_mode : "border" or "zeros"
    """
    return F.grid_sample(inputs, coords, padding_mode = padding_mode, align_corners = align_corners)


def flow_horizontal_flip(flow_input):
    flow_flip = torch.flip(flow_input, [3])
    flow_flip[:, 0:1, :, :] *= -1
    return flow_flip.contiguous()


def post_processing(l_disp, r_disp):
    B, _, H, W = l_disp.shape
    device = l_disp.device
    mean_disp = 0.5 * (l_disp + r_disp) # 평균 disparity를 계산
    
    grid_l = torch.linspace(0.0, 1.0, W).view(1, 1, 1, W).expand(1, 1, H, W).float() # [1, 1, 192, 640]
    grid_l = grid_l.requires_grad_(False).to(device) # 가로로 0에서 1까지 값을 늘려뜨림
    
    # 어떤 기믹인지 잘 모르겠음
    l_mask = 1.0 - torch.clamp(20 * (grid_l - 0.05), min = 0, max = 1)
    r_mask = torch.flip(l_mask, [3])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * mean_disp





####################################################################################################
######## PWC Network for MonoSceneFlow
####################################################################################################
## 코스트 볼륨을 포워드 하기 전에 피처맵 normalization
def feature_normalization(feature_list):
    statistics_mean = []
    statistics_var  = []
    axes = [-3, -2, -1]

    for feature in feature_list:
        statistics_mean.append(feature.mean(dim = axes, keepdims = True))
        statistics_var.append(feature.var(dim = axes, keepdims = True))

    statistics_std = [torch.sqrt(var + 1e-16) for var in statistics_var]
    feature_list   = [feature - mean for feature, mean in zip(feature_list, statistics_mean)]
    feature_list   = [feature / std for feature, std in zip(feature_list, statistics_std)]
    return feature_list



def conv(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1, use_relu = True):
    if use_relu:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation,
                padding = ((kernel_size - 1) * dilation) // 2, bias = True),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation,
                padding = ((kernel_size - 1) * dilation) // 2, bias = True))



class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)



class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs   = nn.ModuleList()
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out))
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)
        return feature_pyramid[::-1]



class MonoSceneFlowDecoder(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowDecoder, self).__init__()
        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32))
        self.conv_sf = conv(32, 3, use_relu=False)
        self.conv_d1 = conv(32, 1, use_relu=False)


    def forward(self, x):
        x_out = self.convs(x)
        sf    = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)
        return x_out, sf, disp1



class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1))
        self.conv_sf = conv(32, 3, use_relu=False)
        self.conv_d1 = nn.Sequential(
            conv(32, 1, use_relu=False), 
            torch.nn.Sigmoid())


    def forward(self, x):
        x_out = self.convs(x)
        sf    = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out) * 0.3
        return sf, disp1



# class CostVolume(nn.Module):
#     def __init__(self):
#         super(CostVolume, self).__init__()
        
        
#     def forward(self, feat1, feat2, param_dict):
#         """
#         only implemented for:
#             kernel_size = 1
#             stride1 = 1
#             stride2 = 1
#         """
#         _, _, H, W   = feat1.size()
#         max_disp     = param_dict["max_disp"]
#         num_shifts   = 2 * max_disp + 1
#         feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

#         cost_list = []
#         for i in range(num_shifts):
#             for j in range(num_shifts):
#                 corr = torch.mean(feat1 * feat2_padded[:, :, i:(H + i), j:(W + j)], axis = 1, keepdims = True)
#                 cost_list.append(corr)
#         cost_volume = torch.cat(cost_list, axis = 1)
#         return cost_volume



def upsample_outputs_as(input_list, ref_list):
    output_list = []
    for index in range(0, len(input_list)):
        output_list.append(interpolate_as(input_list[index], ref_list[index]))
    return output_list



class Meshgrid(nn.Module):
    def __init__(self):
        super(Meshgrid, self).__init__()
        self.width  = 0
        self.height = 0
        self.register_buffer("xx", torch.zeros(1,1))
        self.register_buffer("yy", torch.zeros(1,1))
        self.register_buffer("rangex", torch.zeros(1,1))
        self.register_buffer("rangey", torch.zeros(1,1))


    def _compute_meshgrid(self, width, height):
        torch.arange(0, width, out=self.rangex)
        torch.arange(0, height, out=self.rangey)
        self.xx = self.rangex.repeat(height, 1).contiguous()
        self.yy = self.rangey.repeat(width, 1).t().contiguous()


    def forward(self, width, height):
        if self.width != width or self.height != height:
            self._compute_meshgrid(width=width, height=height)
            self.width = width
            self.height = height
        return self.xx, self.yy



## B, H, W를 이용해서 픽셀 그리드를 얻는 함수
def meshgrid(B, H, W, device):
    grid_h = torch.linspace(0.0, W - 1, W).view(1, 1, 1, W).expand(B, 1, H, W) # 가로
    grid_v = torch.linspace(0.0, H - 1, H).view(1, 1, H, 1).expand(B, 1, H, W) # 세로
    ones   = torch.ones_like(grid_h) # 호모지니어스 좌표

    # (X, Y, 1) 좌표를 채널 dim으로 스택
    pixel_grid = torch.cat((grid_h, grid_v, ones), dim = 1).float()
    pixel_grid = pixel_grid.requires_grad_(False).to(device)
    return pixel_grid



## (KITTI 데이터) disp -> depth 변환 함수
def disp2depth_stereo_clamp(pred_disp, k_value):
    pred_depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-8)
    pred_depth = torch.clamp(pred_depth, 1e-3, 80)
    return pred_depth



## (KITTI 데이터) depth -> disp 변환 함수
def depth2disp_stereo(depth, k_value):
    disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth
    return disp



## 뎁스를 이용해서 3차원 포인트 클라우드를 계산
def pixel2point(intrinsic, depth):
    """
    intrinsic와 depth 정보로 해당 프레임의 point cloud를 복원
    1. depth로 pixelgrid 생성
    2. pixelgrid에 inv_intrinsic으로 normlaized 좌표를 만들고,
       뎁스값을 곱해서 포인트 클라우드 좌표를 만듬
    """
    device = intrinsic.device
    B, _, H, W  = depth.size()
    depth_mat   = depth.view(B, 1, -1)

    pixelgrid   = meshgrid(B, H, W, device)
    pixel_mat   = pixelgrid.view(B, 3, -1)

    point_cloud = torch.matmul(torch.inverse(intrinsic.cpu()).to(device), pixel_mat) * depth_mat
    point_cloud = point_cloud.view(B, -1, H, W)
    return point_cloud, pixelgrid



## 포인트 클라우드와 intrinsic으로 픽셀 그리드를 계산
def point2pixel(intrinsic, point):
    """
    포인트 클라우드 [X, Y, Z]를 받아서 intrinsic을 통해 pixel 좌표로 사영하는 함수
    """
    B, _, H, W = point.size()
    proj_point = torch.matmul(intrinsic, point.view(B, 3, -1))
    pixels_mat = proj_point.div(proj_point[:, 2:3, :] + 1e-8)[:, 0:2, :]
    return pixels_mat.view(B, 2, H ,W)



## intinsic, disp, relavite_scale을 이용해서 픽셀 -> 포인트 계산
def pixel2point_ms(intrinsic, pred_disp, rel_scale):
    """
    intrinsic은 aug_size (192, 640)의 스케일, pred_disp는 (192, 640) 부터 (24, 80) 까지 4개의 멀티 스케일
    따라서 intrinsic을 rescaling 해주어야 함
    rescaling한 intrisic과 disp와 focal_length fx로 depth를 복원
    depth와 rescaling_intrinsic으로 포인트 클라우드를 복원
    """
    rescale_K  = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
    pred_depth = disp2depth_stereo_clamp(pred_disp, rescale_K[:, 0, 0])
    point, _   = pixel2point(rescale_K, pred_depth)
    return point, rescale_K



## intrinsic, 포인트 클라우드, disp_size을 이용해서 포인트 -> 픽셀 사영 함수
def point2pixel_ms(intrinsic, point, scene, disp_size):
    """
    예시
    l1 이미지 뎁스 disp_l1의 depth로 복원한 point에 Sfw를 더하면 l2 뷰의 포인트 클라우드가 됨
    l2 이미지 뎁스 disp_l2의 depth로 복원한 point에 Sbw를 더하면 l1 뷰의 포인트 클라우드가 됨
    """
    rescale_scene = F.interpolate(scene, disp_size, mode = "bilinear", align_corners = True)
    point_form    = point + rescale_scene

    
    coord         = point2pixel(intrinsic, point_form)
    norm_coord_w  = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
    norm_coord_h  = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
    norm_coord    = torch.cat((norm_coord_w, norm_coord_h), dim = 1)
    return rescale_scene, point_form, norm_coord



## 좌표를 이용해서 이미지를 와핑하는 함수
def recon_image(coord, image):
    """
    변환 좌표와 이미지를 받아서 와핑하는 함수
    """
    grid     = coord.transpose(1, 2).transpose(2, 3)
    img_warp = F.grid_sample(image, grid, align_corners = True)

    mask = torch.ones_like(image, requires_grad = False)
    mask = F.grid_sample(mask, grid, align_corners = True)
    mask = (mask >= 1.0).float()
    return img_warp * mask



## 좌표를 이용해서 포인트 클라우드를 와핑하는 함수
def recon_point(coord, point):
    """
    변환 좌표와 포인트 클라우드르 받아서 와핑하는 함수
    """
    grid       = coord.transpose(1, 2).transpose(2, 3)
    point_warp = F.grid_sample(point, grid, align_corners = True)

    mask = torch.ones_like(point, requires_grad = False)
    mask = F.grid_sample(mask, grid, align_corners = True)
    mask = (mask >= 1.0).float()
    return point_warp * mask



## Sceneflow를 Opticalflow로 사영하는 함수
def projectScene2Flow(intrinsic, scene, disp, flow):
    B, C, H, W    = disp.size()
    ## rescale disp -> pred absolute depth
    depth         = disp2depth_stereo_clamp(disp, intrinsic[:, 0, 0])
    ## pred absolute depth -> pred absolute point cloud
    point, grid   = pixel2point(intrinsic, depth)
    ## scene flow를 disp 사이즈로 interpolation하고 point에 더해서 교정한 point를 계산
    rescale_scene = F.interpolate(scene, [H, W], mode = "bilinear", align_corners = True)



    # disp -> depth -> point에 rescale_scene을 더해서 인접 프레임의 뷰의 포인트로 계산
    point_form     = point + rescale_scene
    # print("point cloud   ")
    # print(point.view(B, 3, -1)[:, :, :5])
    # print("rescale_scene  ")
    # print(rescale_scene.view(B, 3, -1)[:, :, :5])

    # print("intrinsic     ")
    # print(intrinsic)
    # print("point form    ")
    # print(point_form.view(B, 3, -1)[:, :, :5])
    # print("proj point    ")
    # print(torch.matmul(intrinsic, point_form.view(B, 3, -1))[:, :, :5])
    coord          = point2pixel(intrinsic, point_form)
    optical_flow   = coord - grid[:, 0:2, :, :]
    # print("optical flow")
    # print(optical_flow.view(B, 2, -1)[:, :, :5])
    # print("gt flow")
    # print(flow.view(B, 2, -1)[:, :, :5])
    return optical_flow



## 원본 스케일의 intrisic에 스케일 팩터를 곱해서 스케일링하는 함수
def intrinsic_scale(intrinsic, scale_y, scale_x):
    device  = intrinsic.device
    B, H, W = intrinsic.size()
    fx = intrinsic[:, 0, 0] * scale_x
    fy = intrinsic[:, 1, 1] * scale_y
    cx = intrinsic[:, 0, 2] * scale_x
    cy = intrinsic[:, 1, 2] * scale_y

    zeros = torch.zeros_like(fx)
    r1 = torch.stack([fx, zeros, cx], dim = 1)
    r2 = torch.stack([zeros, fy, cy], dim = 1)
    r3 = torch.tensor([0., 0., 1.], requires_grad = False).to(device).unsqueeze(0).expand(B, -1)
    rescale_K = torch.stack([r1, r2, r3], dim=1)
    return rescale_K



class WarpingSceneFlow(nn.Module):
    def __init__(self):
        super(WarpingSceneFlow, self).__init__()
        """
        disp와 intrinsic으로 point cloud를 계산
        point cloud에 scene flow을 더해서 인접 프레임의 scene flow point cloud로 옮김
        이 포인트 클라우드를 해당 프레임으로 projection하면 WarpingSceneFlow
        """
    def forward(self, image, disp, scene, intrinsic, aug_size):
        B, C, H, W  = image.size()
        disp = interpolate_as(disp, image) * W
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = H
        local_scale[:, 1] = W
        
        point, scaeld_k = pixel2point_ms(intrinsic, disp, local_scale / aug_size)
        _, _, coord     = point2pixel_ms(scaeld_k, point, scene, [H, W])

        grid = coord.transpose(1, 2).transpose(2, 3)
        warp = F.grid_sample(image, grid, align_corners = True)
        mask = torch.ones_like(image, requires_grad = False)
        mask = F.grid_sample(mask, grid, align_corners = True)
        mask = (mask >= 1.0).float()
        return warp