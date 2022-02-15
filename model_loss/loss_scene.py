import torch
import torch.nn as nn
import torch.nn.functional as F
from model_layer import *



##############################################################################################
## Basic Module (SSIM + Smooth 2nd)
##############################################################################################
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.C1   = 0.01 ** 2
        self.C2   = 0.03 ** 2
    

    def forward(self, x, y):
        mu_x      = nn.AvgPool2d(3, 1)(x)
        mu_y      = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq   = mu_x.pow(2)
        mu_y_sq   = mu_y.pow(2)
        sigma_x   = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y   = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy  = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n    = (2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d    = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM      = SSIM_n / SSIM_d
        SSIM_img  = torch.clamp((1 - SSIM) / 2, 0, 1)
        return F.pad(SSIM_img, pad = (1, 1, 1, 1), mode = 'constant', value = 0)



class Smooth2nd(nn.Module):
    def __init__(self):
        super(Smooth2nd, self).__init__()
        """
        2-order smoothness 로스 계산
        """
    def forward(self, scene, imag, beta):
        sf_grad_x  = gradient_x_2nd(scene)
        sf_grad_y  = gradient_y_2nd(scene)

        img_grad_x = gradient_x(imag) 
        img_grad_y = gradient_y(imag) 
        weights_x  = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
        weights_y  = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

        smoothness_x = sf_grad_x * weights_x
        smoothness_y = sf_grad_y * weights_y
        return (smoothness_x.abs() + smoothness_y.abs())



##############################################################################################
## Loss function
##############################################################################################
class SelfSceneFlowLoss(nn.Module):
    def __init__(self):
        super(SelfSceneFlowLoss, self).__init__()
        self.ssim    = SSIM()
        self.smooth  = Smooth2nd()
        self.weights      = [4.0, 2.0, 1.0, 1.0, 1.0]
        self.disp_ssim    = 0.85
        self.disp_smooth  = 0.1
        self.scene_point  = 0.2
        self.scene_smooth = 200
        # self.image = []
        # self.scene = []


    def apply_disparity(self, image, disp):
        """
        Args:
            image: [B, 3, H, W], 이미지 RGB 값
            -disp: [B, 1, H, W], 0 ~ 1에서 -1 ~ 0 범위로 바뀜

        오피셜 코드는 0.3 * sigmoid이고 2 * flow - 1 텀을 사용 -> 이것이 있어야 와핑이 올바르게 됨
        """
        B, _, H, W = image.size()
        ## Original coordinates of pixels, 범위는 W 방향으로, H 방향으로 0 .... 1
        ## 인덱스로 [0, 0] 에서 0, 0, [W, H]에서 [1, 1]
        x_base = torch.linspace(0, 1, W).repeat(B, H, 1).type_as(image)
        y_base = torch.linspace(0, 1, H).repeat(B, W, 1).transpose(1, 2).type_as(image)

        ## 음수 디스패리티를 x base에 더해서 L 좌표계를 만듬
        ## flow_field의 범위는 0 ~ 1 + 0 + -1 이므로 범위는 -1 ~ 1이 됨
        x_shifts   = disp[:, 0, :, :] 
        flow_field = torch.stack((x_base + x_shifts, y_base), dim = 3)

        ## 그리드 샘플의 좌표 범위를 -1 에서 1로 제한 == 2*flow_field - 1
        ## 2 * flow -1 항이 있어야 warping이 제대로 됨
        output = F.grid_sample(
            image, 2*flow_field-1, mode = "bilinear", padding_mode = "zeros", align_corners = True)
        return output


    def monodepth_loss(self, disp_l, disp_r, img_l_aug, img_r_aug, index):
        """
        우측 이미지를 -disp_l을 이용해서 좌측 이미지의 좌표로 와핑
        이 과정을 r1_aug -> l1, r2_aug -> l2로 수행
        """
        img_r_warp  = self.apply_disparity(img_r_aug, -disp_l) ## 우측 이미지를 좌측 이미지로 와핑
        ## outputs[("depth_warp_r", index)] = img_r_warp

        ## Photometric loss == |a-b| + SSIM(a, b)
        img_diff    = ((1.0 - self.disp_ssim) * elementwise_l1(img_l_aug, img_r_warp)).mean(dim=1, keepdim=True)
        img_diff    += (self.disp_ssim * self.ssim(img_l_aug, img_r_warp)).mean(dim=1, keepdim=True)
        loss_img    = img_diff.mean()

        ## Disparities smoothness
        loss_smooth = self.smooth(disp_l, img_l_aug, beta = 10.0).mean() / (2 ** index)
        return loss_img + self.disp_smooth * loss_smooth


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l_aug, img_l1_aug, img_l2_aug, aug_size, index):
        _, _, H, W = sf_f.size() # [B, 3, 192, 640]
        disp_l1 = disp_l1 * W    # 왜 곱해줄까?
        disp_l2 = disp_l2 * W    # 왜 곱하는지 알겠다. 시차의 범위는 0에서 최대 W까지기 때문

        ## scale
        local_scale = torch.zeros_like(aug_size) # aug_size = 192, 640
        local_scale[:, 0] = H
        local_scale[:, 1] = W         

        ## 입력은 원본 intrinsic, disp per scale, 현재 스케일/원본 스케일 비율
        ## 출력은 포인트 클라우드 per scale, rescaling intrinsic matrix
        point1, k1_scale    = pixel2point_ms(k_l_aug, disp_l1, local_scale / aug_size)
        point2, k2_scale    = pixel2point_ms(k_l_aug, disp_l2, local_scale / aug_size)

        ## 입력은 rescaling intrinsic, 해당 프레임의 포인트 클라우드, 해당 프레임 -> 인접 프레임의 scene flow
        ## 출력은 l1 좌표계에서 바라본 point1 + sf_f 좌표 (l2 좌표로 와핑한 것은 아님) 와 이를 l2 좌표계로 사영한 그리드
        ## 출력은 l2 좌표계에서 바라본 point2 + sf_b 좌표와 l1 좌표계로 사영한 그리드 (coord 좌표계 노테이션에 주의)
        _, point1_f, coord1 = point2pixel_ms(k1_scale, point1, sf_f, [H, W])
        _, point2_f, coord2 = point2pixel_ms(k2_scale, point2, sf_b, [H, W]) 


        ## 1. l1으로부터 계산한 와핑 좌표 coord1에 l2 이미지를 와핑해서 l1 뷰로 합셩험
        ## 2. l2으로부터 계산한 와핑 좌표 coord2에 l1 이미지를 와핑해서 l2 뷰로 합성함
        img_l2_warp = recon_image(coord1, img_l2_aug) ## l2를 와핑한 것은 l1와 같아야힘
        img_l1_warp = recon_image(coord2, img_l1_aug) ## l1을 와핑한 것은 l2와 같아야함
        ## outputs[("scene_warp_l1", index)] = img_l2_warp
        ## outputs[("scene_warp_l2", index)] = img_l1_warp
        ## self.image.append(img_l2_aug)
        ## print(coord1)
        ## self.scene.append(img_l2_warp)

        ## 3. l2의 포인트 클라우드를 인접 좌표계 coord1 와핑
        ## 4. l1의 포인트 클라우드를 인접 좌표계 coord2 와핑
        point2_warp = recon_point(coord1, point2)
        point1_warp = recon_point(coord2, point1) 


        ################# Image reconstruction loss
        ## self.outputs.append(img_l2_warp)
        img_diff1   = ((1.0 - self.disp_ssim) * elementwise_l1(img_l1_aug, img_l2_warp)).mean(dim=1, keepdim=True)
        img_diff1   += (self.disp_ssim * self.ssim(img_l1_aug, img_l2_warp)).mean(dim=1, keepdim=True)
        img_diff2   = ((1.0 - self.disp_ssim) * elementwise_l1(img_l2_aug, img_l1_warp)).mean(dim=1, keepdim=True)
        img_diff2   += (self.disp_ssim * self.ssim(img_l2_aug, img_l1_warp)).mean(dim=1, keepdim=True)
        loss_image  = img_diff1.mean() + img_diff2.mean() ## scene flow
        
        ################# Point reconstruction Loss
        point_norm1 = torch.norm(point1, p = 2, dim = 1, keepdim = True)
        point_norm2 = torch.norm(point2, p = 2, dim = 1, keepdim = True)
        point_diff1 = elementwise_epe(point1_f, point2_warp).mean(dim = 1, keepdim = True) / (point_norm1 + 1e-8)
        point_diff2 = elementwise_epe(point2_f, point1_warp).mean(dim = 1, keepdim = True) / (point_norm2 + 1e-8)
        loss_point  = point_diff1.mean() + point_diff2.mean()

        ################# 3D motion smoothness loss
        loss_3d_smooth = (self.smooth(sf_f, img_l1_aug, beta=10.0) / (point_norm1 + 1e-8)).mean() / (2 ** index)
        loss_3d_smooth += (self.smooth(sf_b, img_l2_aug, beta=10.0) / (point_norm2 + 1e-8)).mean() / (2 ** index)

        ################# Loss Summnation
        sceneflow_loss = loss_image + self.scene_point * loss_point + self.scene_smooth * loss_3d_smooth
        return sceneflow_loss, loss_image, loss_point, loss_3d_smooth


    def graidnet_detach(self, output_dict):
        for index in range(0, len(output_dict['scene_f'])):
            output_dict['scene_f'][index].detach_()
            output_dict['scene_b'][index].detach_()
            output_dict['disp_l1'][index].detach_()
            output_dict['disp_l2'][index].detach_()
        return None


    def forward(self, target_dict, output_dict):
        loss_dict    = {}
        loss_dp_sum  = 0
        loss_sf_sum  = 0
        loss_sf_2d   = 0
        # loss_sf_3d   = 0
        loss_sf_sm   = 0
        
        ## K_l, K_r하고 output_dicr_r의 disp_l1, disp_l2 얻음
        K_l_aug      = target_dict["kl_aug"]
        aug_size     = target_dict['aug_size']
        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for index, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(
            zip(output_dict['scene_f'], output_dict['scene_b'], 
                output_dict['disp_l1'], output_dict['disp_l2'], 
                disp_r1_dict, disp_r2_dict)):
                
            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            ## print(torch.max(disp_l1), torch.min(disp_l1))
            ## l1, l2, r1, r2 이미지를 scene flow 스케일에 맞게 interpolation
            img_l1_aug = interpolate_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate_as(target_dict["input_r2_aug"], sf_b)

            ## Disparity Loss (L1 + SSIM)
            loss_disp_l1 = self.monodepth_loss(disp_l1, disp_r1, img_l1_aug, img_r1_aug, index)
            loss_disp_l2 = self.monodepth_loss(disp_l2, disp_r2, img_l2_aug, img_r2_aug, index)
            loss_dp_sum  = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self.weights[index]

            ## Sceneflow Loss           
            # loss_sceneflow, loss_image, loss_point, loss_3d_smooth = self.sceneflow_loss(
            #     sf_f, sf_b, disp_l1, disp_l2, K_l_aug, img_l1_aug, img_l2_aug, aug_size, index)
            loss_sceneflow, loss_image, loss_3d_smooth = self.sceneflow_loss(
                sf_f, sf_b, disp_l1, disp_l2, K_l_aug, img_l1_aug, img_l2_aug, aug_size, index)
            loss_sf_sum = loss_sf_sum + loss_sceneflow * self.weights[index]            
            loss_sf_2d  = loss_sf_2d + loss_image            
            # loss_sf_3d  = loss_sf_3d + loss_point
            loss_sf_sm  = loss_sf_sm + loss_3d_smooth

        ## finding weight
        f_loss     = loss_sf_sum.detach()
        d_loss     = loss_dp_sum.detach()
        max_val    = torch.max(f_loss, d_loss)
        f_weight   = max_val / f_loss
        d_weight   = max_val / d_loss
        total_loss = loss_dp_sum * d_weight + loss_sf_sum * f_weight

        loss_dict  = {}
        loss_dict["dp"]    = loss_dp_sum
        loss_dict["sf"]    = loss_sf_sum
        loss_dict["s2"]    = loss_sf_2d
        # loss_dict["s3"]    = loss_sf_3d
        loss_dict["s3sm"]  = loss_sf_sm
        loss_dict["total"] = total_loss

        ## 우측 이미지는 로스 가이던스로만 사용해, 그래서 grdient detach를 수행
        self.graidnet_detach(output_dict['output_dict_r'])
        return loss_dict



# class SelfSceneFlowLoss_NotPoint(nn.Module):
#     def __init__(self):
#         super(SelfSceneFlowLoss_NotPoint, self).__init__()
#         self.ssim    = SSIM()
#         self.smooth  = Smooth2nd()
#         self.weights      = [4.0, 2.0, 1.0, 1.0, 1.0]
#         self.disp_ssim    = 0.85
#         self.disp_smooth  = 0.1
#         self.scene_point  = 0.2
#         self.scene_smooth = 200
#         # self.image = []
#         # self.scene = []


#     def apply_disparity(self, image, disp):
#         """
#         Args:
#             image: [B, 3, H, W], 이미지 RGB 값
#             -disp: [B, 1, H, W], 0 ~ 1에서 -1 ~ 0 범위로 바뀜

#         오피셜 코드는 0.3 * sigmoid이고 2 * flow - 1 텀을 사용 -> 이것이 있어야 와핑이 올바르게 됨
#         """
#         B, _, H, W = image.size()
#         ## Original coordinates of pixels, 범위는 W 방향으로, H 방향으로 0 .... 1
#         ## 인덱스로 [0, 0] 에서 0, 0, [W, H]에서 [1, 1]
#         x_base = torch.linspace(0, 1, W).repeat(B, H, 1).type_as(image)
#         y_base = torch.linspace(0, 1, H).repeat(B, W, 1).transpose(1, 2).type_as(image)

#         ## 음수 디스패리티를 x base에 더해서 L 좌표계를 만듬
#         ## flow_field의 범위는 0 ~ 1 + 0 + -1 이므로 범위는 -1 ~ 1이 됨
#         x_shifts   = disp[:, 0, :, :] 
#         flow_field = torch.stack((x_base + x_shifts, y_base), dim = 3)

#         ## 그리드 샘플의 좌표 범위를 -1 에서 1로 제한 == 2*flow_field - 1
#         ## 2 * flow -1 항이 있어야 warping이 제대로 됨
#         output = F.grid_sample(
#             image, 2*flow_field-1, mode = "bilinear", padding_mode = "zeros", align_corners = True)
#         return output


#     def monodepth_loss(self, disp_l, disp_r, img_l_aug, img_r_aug, index):
#         """
#         우측 이미지를 -disp_l을 이용해서 좌측 이미지의 좌표로 와핑
#         이 과정을 r1_aug -> l1, r2_aug -> l2로 수행
#         """
#         img_r_warp  = self.apply_disparity(img_r_aug, -disp_l) ## 우측 이미지를 좌측 이미지로 와핑
#         ## outputs[("depth_warp_r", index)] = img_r_warp

#         ## Photometric loss == |a-b| + SSIM(a, b)
#         img_diff    = ((1.0 - self.disp_ssim) * elementwise_l1(img_l_aug, img_r_warp)).mean(dim=1, keepdim=True)
#         img_diff    += (self.disp_ssim * self.ssim(img_l_aug, img_r_warp)).mean(dim=1, keepdim=True)
#         loss_img    = img_diff.mean()

#         ## Disparities smoothness
#         loss_smooth = self.smooth(disp_l, img_l_aug, beta = 10.0).mean() / (2 ** index)
#         return loss_img + self.disp_smooth * loss_smooth


#     def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l_aug, img_l1_aug, img_l2_aug, aug_size, index):
#         _, _, H, W = sf_f.size() # [B, 3, 192, 640]
#         disp_l1 = disp_l1 * W    # 왜 곱해줄까?
#         disp_l2 = disp_l2 * W    # 왜 곱하는지 알겠다. 시차의 범위는 0에서 최대 W까지기 때문

#         ## scale
#         local_scale = torch.zeros_like(aug_size) # aug_size = 192, 640
#         local_scale[:, 0] = H
#         local_scale[:, 1] = W         

#         ## 입력은 원본 intrinsic, disp per scale, 현재 스케일/원본 스케일 비율
#         ## 출력은 포인트 클라우드 per scale, rescaling intrinsic matrix
#         point1, k1_scale    = pixel2point_ms(k_l_aug, disp_l1, local_scale / aug_size)
#         point2, k2_scale    = pixel2point_ms(k_l_aug, disp_l2, local_scale / aug_size)

#         ## 입력은 rescaling intrinsic, 해당 프레임의 포인트 클라우드, 해당 프레임 -> 인접 프레임의 scene flow
#         ## 출력은 l1 좌표계에서 바라본 point1 + sf_f 좌표 (l2 좌표로 와핑한 것은 아님) 와 이를 l2 좌표계로 사영한 그리드
#         ## 출력은 l2 좌표계에서 바라본 point2 + sf_b 좌표와 l1 좌표계로 사영한 그리드 (coord 좌표계 노테이션에 주의)
#         _, point1_f, coord1 = point2pixel_ms(k1_scale, point1, sf_f, [H, W])
#         _, point2_f, coord2 = point2pixel_ms(k2_scale, point2, sf_b, [H, W]) 


#         ## 1. l1으로부터 계산한 와핑 좌표 coord1에 l2 이미지를 와핑해서 l1 뷰로 합셩험
#         ## 2. l2으로부터 계산한 와핑 좌표 coord2에 l1 이미지를 와핑해서 l2 뷰로 합성함
#         img_l2_warp = recon_image(coord1, img_l2_aug) ## l2를 와핑한 것은 l1와 같아야힘
#         img_l1_warp = recon_image(coord2, img_l1_aug) ## l1을 와핑한 것은 l2와 같아야함
#         ## outputs[("scene_warp_l1", index)] = img_l2_warp
#         ## outputs[("scene_warp_l2", index)] = img_l1_warp
#         ## self.image.append(img_l2_aug)
#         ## print(coord1)
#         ## self.scene.append(img_l2_warp)


#         ################# Image reconstruction loss
#         ## self.outputs.append(img_l2_warp)
#         img_diff1   = ((1.0 - self.disp_ssim) * elementwise_l1(img_l1_aug, img_l2_warp)).mean(dim=1, keepdim=True)
#         img_diff1   += (self.disp_ssim * self.ssim(img_l1_aug, img_l2_warp)).mean(dim=1, keepdim=True)
#         img_diff2   = ((1.0 - self.disp_ssim) * elementwise_l1(img_l2_aug, img_l1_warp)).mean(dim=1, keepdim=True)
#         img_diff2   += (self.disp_ssim * self.ssim(img_l2_aug, img_l1_warp)).mean(dim=1, keepdim=True)
#         loss_image  = img_diff1.mean() + img_diff2.mean() ## scene flow
        
#         point_norm1 = torch.norm(point1, p = 2, dim = 1, keepdim = True)
#         point_norm2 = torch.norm(point2, p = 2, dim = 1, keepdim = True)

#         ################# 3D motion smoothness loss
#         loss_3d_smooth = (self.smooth(sf_f, img_l1_aug, beta=10.0) / (point_norm1 + 1e-8)).mean() / (2 ** index)
#         loss_3d_smooth += (self.smooth(sf_b, img_l2_aug, beta=10.0) / (point_norm2 + 1e-8)).mean() / (2 ** index)

#         ################# Loss Summnation
#         sceneflow_loss = loss_image + self.scene_smooth * loss_3d_smooth
#         return sceneflow_loss, loss_image, loss_3d_smooth


#     def graidnet_detach(self, output_dict):
#         for index in range(0, len(output_dict['scene_f'])):
#             output_dict['scene_f'][index].detach_()
#             output_dict['scene_b'][index].detach_()
#             output_dict['disp_l1'][index].detach_()
#             output_dict['disp_l2'][index].detach_()
#         return None


#     def forward(self, target_dict, output_dict):
#         loss_dict    = {}
#         loss_dp_sum  = 0
#         loss_sf_sum  = 0
#         loss_sf_2d   = 0
#         loss_sf_sm   = 0
        
#         ## K_l, K_r하고 output_dicr_r의 disp_l1, disp_l2 얻음
#         K_l_aug      = target_dict["kl_aug"]
#         aug_size     = target_dict['aug_size']
#         disp_r1_dict = output_dict['output_dict_r']['disp_l1']
#         disp_r2_dict = output_dict['output_dict_r']['disp_l2']

#         for index, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(
#             zip(output_dict['scene_f'], output_dict['scene_b'], 
#                 output_dict['disp_l1'], output_dict['disp_l2'], 
#                 disp_r1_dict, disp_r2_dict)):
                
#             assert(sf_f.size()[2:4] == sf_b.size()[2:4])
#             assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
#             assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
#             ## print(torch.max(disp_l1), torch.min(disp_l1))
#             ## l1, l2, r1, r2 이미지를 scene flow 스케일에 맞게 interpolation
#             img_l1_aug = interpolate_as(target_dict["input_l1_aug"], sf_f)
#             img_l2_aug = interpolate_as(target_dict["input_l2_aug"], sf_b)
#             img_r1_aug = interpolate_as(target_dict["input_r1_aug"], sf_f)
#             img_r2_aug = interpolate_as(target_dict["input_r2_aug"], sf_b)

#             ## Disparity Loss (L1 + SSIM)
#             loss_disp_l1 = self.monodepth_loss(disp_l1, disp_r1, img_l1_aug, img_r1_aug, index)
#             loss_disp_l2 = self.monodepth_loss(disp_l2, disp_r2, img_l2_aug, img_r2_aug, index)
#             loss_dp_sum  = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self.weights[index]

#             loss_sceneflow, loss_image, loss_3d_smooth = self.sceneflow_loss(
#                 sf_f, sf_b, disp_l1, disp_l2, K_l_aug, img_l1_aug, img_l2_aug, aug_size, index)
#             loss_sf_sum = loss_sf_sum + loss_sceneflow * self.weights[index]            
#             loss_sf_2d  = loss_sf_2d + loss_image            
#             loss_sf_sm  = loss_sf_sm + loss_3d_smooth

#         ## finding weight
#         f_loss     = loss_sf_sum.detach()
#         d_loss     = loss_dp_sum.detach()
#         max_val    = torch.max(f_loss, d_loss)
#         f_weight   = max_val / f_loss
#         d_weight   = max_val / d_loss
#         total_loss = loss_dp_sum * d_weight + loss_sf_sum * f_weight

#         loss_dict  = {}
#         loss_dict["dp"]    = loss_dp_sum
#         loss_dict["sf"]    = loss_sf_sum
#         loss_dict["s2"]    = loss_sf_2d
#         loss_dict["s3sm"]  = loss_sf_sm
#         loss_dict["total"] = total_loss

#         ## 우측 이미지는 로스 가이던스로만 사용해, 그래서 grdient detach를 수행
#         self.graidnet_detach(output_dict['output_dict_r'])
#         return loss_dict