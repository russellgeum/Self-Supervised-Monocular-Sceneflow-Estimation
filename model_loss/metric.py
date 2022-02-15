import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_layer import *
from .loss_scene import *



def compute_depth_error(ground_truth, prediction, lib = "torch"):
    """
    추정한 뎁스와 GT의 차이를 계산, cuda을 물은 텐서를 GPU에서 계산 -> cpu().item()으로 cuda 해제해서 리턴
    Args: 
        ground_truth: [B, 1, H, W]
        prediction:   [B, 1, H, W]
    """
    if lib == "numpy":
        thresh   = np.maximum((ground_truth / prediction), (prediction / ground_truth))
        a1       = (thresh < 1.25     ).mean()
        a2       = (thresh < 1.25 ** 2).mean()
        a3       = (thresh < 1.25 ** 3).mean()

        rmse     = (ground_truth - prediction) ** 2
        rmse     = np.sqrt(rmse.mean())

        rmse_log = (np.log(ground_truth) - np.log(prediction)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel  = np.mean(np.abs(ground_truth - prediction) / ground_truth)
        sqrt_rel = np.mean(((ground_truth - prediction) ** 2) / ground_truth)

    elif lib == "torch":
        """
        추정한 뎁스와 GT의 차이를 계산
        Args: 
            ground_truth: [B, 1, H, W]
            prediction: [B, 1, H, W]
        """
        threshold = torch.maximum((ground_truth / prediction), (prediction / ground_truth))
        a1        = (threshold < 1.25     ).float().mean()
        a2        = (threshold < 1.25 ** 2).float().mean()
        a3        = (threshold < 1.25 ** 3).float().mean()

        rmse      = (ground_truth - prediction) ** 2
        rmse      = torch.sqrt(rmse.mean())
        
        rmse_log  = (torch.log(ground_truth) - torch.log(prediction)) ** 2
        rmse_log  = torch.sqrt(rmse_log.mean())

        abs_rel   = torch.mean(torch.abs(ground_truth - prediction) / ground_truth)
        sqrt_rel  = torch.mean((ground_truth - prediction) ** 2 / ground_truth)

        if abs_rel.device is not "cpu": # 메트릭 디바이스가 CPU가 아니면 GPU임
            abs_rel  = abs_rel.cpu().item()
            sqrt_rel = sqrt_rel.cpu().item()
            rmse     = rmse.cpu().item()
            rmse_log = rmse_log.cpu().item()
            a1       = a1.cpu().item()
            a2       = a2.cpu().item()
            a3       = a3.cpu().item()
        else: # 디바이스 cpu라면 item()만 얻어옴
            abs_rel  = abs_rel.item()
            sqrt_rel = sqrt_rel.item()
            rmse     = rmse.item()
            rmse_log = rmse_log.item()
            a1       = a1.item()
            a2       = a2.item()
            a3       = a3.item()
    else:
        raise "lib arg is 'numpy' or 'torch'"
    return [abs_rel, sqrt_rel, rmse, rmse_log, a1, a2, a3]



## 모노뎁스 메트릭, 미디언 스케일링을 사용하지 않음
class MonoDepthEvaluation(nn.Module):
    def __init__(self):
        super(MonoDepthEvaluation, self).__init__()
        self.depth_metric = ["abs_rel", "sqrt_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    

    # MonoScene용 코드
    # https://github.com/visinf/self-mono-sf/blob/master/losses.py#L427
    def disp2depth(self, disp, k_value):
        mask  = (disp > 0).float()
        depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))
        return depth


    def forward(self, inputs, outputs):
        """
        예측한 disp_l1과 타겟 뎁스 depth_l1의 메트릭을 비교 || pred_disp_l1 -> pred_depth_l1으로 변환
        target_depth_l1 : [B, 1, 375, 1242]
        K_l : [192, 640] 사이즈에 맞게 rescaling intrinsic
        pred_disp_l1 : [B, 1, 192, 640]
        """
        metric_dict = {}
        ## Extracting componene
        target_depth_l1 = inputs["depth_l1"] # 나의 구현에서는 depth_l0, depth_l1, depth_l2 ... ...
        intrinsic       = inputs["kl"]
        predict_disp_l1  = outputs["disp_l1_pp"][0]

        # pred_disp를 target_depth 크기로 interpolation하고 disp width를 곱해줌 
        # (이미지 가로 크기만큼 최대 시차를 고려)
        # 그리고 eval_disp2depth에서 disp - depth 공식을 통해 depth를 계산
        predict_disp_l1  = interpolate_as(predict_disp_l1, target_depth_l1, "bilinear", align_corners = True)
        predict_disp_l1  = predict_disp_l1 * target_depth_l1.size(3)
        predict_depth_l1 = self.disp2depth(predict_disp_l1, intrinsic[:, 0, 0])
        predict_depth_l1 = torch.clamp(predict_depth_l1, min = 1e-3, max = 80) ## absolute scale

        # MD2와 다르게 median scaling을 안하는 이유는, stereo로 메트릭을 측정하기 때문에 스케일을 정확하게 알기 때문
        mask = (target_depth_l1 > 1e-3) * (target_depth_l1 < 80.0)
        target_depth_l1  = target_depth_l1[mask]
        predict_depth_l1 = predict_depth_l1[mask]

        metric_list     = compute_depth_error(target_depth_l1, predict_depth_l1)
        # [abs_rel, sqrt_rel, rmse, rmse_log, a1, a2, a3] 메트릭이 포함된 딕셔너리
        for key, value in zip(self.depth_metric, metric_list):
            metric_dict[key] = value
        return metric_dict



class MonoSceneFlowEvaluation(nn.Module): 
    def __init__(self):
        super(MonoSceneFlowEvaluation, self).__init__()
        # selfmono, multimono의 SceneFlow with Depth evaluation
        # 스테레오 이미지로 훈련을 할 때에는 scale을 알 수 있어서, median scaling이 필요하지 않음
        # 그러나 모노큘라 데이터로만 훈련을 할 때에는 scale을 모르기 때문에 median scaling이 필요함
        self.depth_metric = ["abs_rel", "sqrt_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        self.scene_metric = ["d1", "f1", "d2", "sf"]


    # MonoScene용 코드
    # https://github.com/visinf/self-mono-sf/blob/master/losses.py#L427
    def disp2depth_stereo(self, disp, k_value):
        mask  = (disp > 0).float()
        depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))
        return depth


    def depth2disp_stereo(self, depth, k_value):
        disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth
        return disp


    def crop_mask(self, mask):
        crop = torch.zeros_like(mask)
        crop[:, :, 153:371, 44:1197] = 1
        return  mask * crop

    
    def eval_disp_depth(self, target_disp_mask, target_disp, predict_disp, target_depth, predict_depth):
        disp_metric = {}
        B, C, H, W  = target_disp.shape

        # KITTI disparity metric 계산
        # pred_disp와 target_disp 차이의 L2 노름을 계산하고 valid 부분만 mask 씌워서 계산
        disp_valid_epe     = elementwise_epe(predict_disp, target_disp) * target_disp_mask.float() 
        # 3보다 큰 부분과 0.05 보다 큰 부분을 곱하고 mask를 곱해서 유효 부분만 계산
        disp_outliter_epe  = (disp_valid_epe > 3.0).float() * ((disp_valid_epe / target_disp) > 0.05).float()
        disp_outliter_epe  = disp_outliter_epe * target_disp_mask.float()
        # 이 outlier 값은 향후 d1, d2 메트릭으로 쓰일 sla것
        disp_metric["otl"]     = (disp_outliter_epe.view(B, -1).sum(1)).mean() / 91875.68
        disp_metric["otl_img"] = disp_outliter_epe

        # KITTI depth metric 계산
        # target_depth = target_depth[self.crop_mask(target_disp_mask)]
        # predict_depth = predict_depth[self.crop_mask(target_disp_mask)]
        target_depth = target_depth[target_disp_mask]
        predict_depth = predict_depth[target_disp_mask]
        depth_error  = compute_depth_error(target_depth, predict_depth)
        return disp_metric, depth_error


    def forward(self, inputs, outputs):
        scene_dict    = {}
        depth_l1_dict = {}
        depth_l2_dict = {}

        ####################################################################################################
        # 타겟 플로우, 타겟 디스패리티, occlusion, intrinsic 로드
        target_flow        = inputs['target_flow'] #                        [B, 2, 375, 1242]
        target_flow_mask   = (inputs['target_flow_mask']==1).float()#       [B, 1, 375, 1242]
        target_disp1       = inputs['target_disp1'] #                       [B, 1, 375, 1242]
        target_disp1_mask  = (inputs['target_disp1_mask']==1).float() #     [B, 1, 375, 1242]
        target_disp2_occ   = inputs['target_disp2_occ'] #                   [B, 1, 375, 1242]
        target_disp2_mask  = (inputs['target_disp2_mask_occ']==1).float() # [B, 1, 375, 1242]
        # 왜 scene_mask가 이런 형태이지?
        target_scene_mask  = target_flow_mask * target_disp1_mask * target_disp2_mask
        B, C, H, W         = target_disp1.size()
        # outputs에서 예측한 disp_l1와 scene_f를 선택 (post processing 한걸로)
        predict_disp_l1    = outputs["disp_l1_pp"][0]
        predict_scene_f    = outputs["scene_f_pp"][0]        
        intrinsic          = inputs["kl"]                


        ####################################################################################################
        ## D1
        ## disp_l1 -> rescale_disp -> absolute_depth (with torch.clamp 80)
        predict_disp_l1    = interpolate_as(predict_disp_l1, target_disp1, mode = "bilinear") * W ## rescale disp
        predict_depth_l1   = self.disp2depth_stereo(predict_disp_l1, intrinsic[:, 0, 0]) ## absolute depth
        predict_depth_l1   = torch.clamp(predict_depth_l1, 1e-3, 80) ## clamped pred absolute depth
        ## gt_disp_l1 -> gt_absolute_depth
        target_depth_l1    = self.disp2depth_stereo(target_disp1, intrinsic[:, 0, 0])

        ## L1 이미지에 대해서 disparity outlier 메트릭과 depth metric 계산
        disp_l1_metric, depth_l1_metric = self.eval_disp_depth(
            target_disp1_mask.bool(), target_disp1, predict_disp_l1, target_depth_l1, predict_depth_l1)
        depth_l1_dict.update({key: depth_l1_metric[index] for index, key in enumerate(self.depth_metric)})

        scene_dict["d1"]            = disp_l1_metric["otl"]
        disp_l1_otl_img             = disp_l1_metric["otl_img"]
        outputs["otl1_img"]         = disp_l1_otl_img
        outputs["predict_disp_l1"]  = predict_disp_l1
        outputs["predict_depth_l1"] = predict_depth_l1


        ####################################################################################################
        ## F1
        ## Scene flow를 target flow 사이즈로 resize 후, K, disp를 이용해서 opticalflow로 계산
        ## [B, 1, 192, 640] -> [B, 1, 375, 1242]로 interpolation
        predict_sceneflow   = interpolate_as(predict_scene_f, target_flow, mode = "bilinear")
        predict_opticalflow = projectScene2Flow(intrinsic, predict_sceneflow, predict_disp_l1, target_flow)
        # target flow와 optical flow의 차이를 계산하고, valid_mask를 씌워서 유효 부분만 필터링
        valid_epe           = elementwise_epe(predict_opticalflow, target_flow) * target_flow_mask
        outputs["optical_flow"] = predict_opticalflow
        
        # target_flow를 L2 norm 취한 값을 mag로 두고, f1 메트릭 계산
        target_flow_mag     = torch.norm(target_flow, p = 2, dim = 1, keepdim = True) + 1e-8
        flow_outlier_epe    = (valid_epe > 3).float() * ((valid_epe / target_flow_mag) > 0.05).float() * target_flow_mask
        scene_dict["f1"]    = (flow_outlier_epe.view(B, -1).sum(1)).mean() / 91875.68


        ####################################################################################################
        ## D2
        ## depth_l1에 predict_sceneflow의 Z 성분을 더해서 depth_l1_nexr를 계산 (이상적으론 depth_l2 와 같아야 할 것)
        ## predict_depth_l1: [B, 1, 375, 1242], predict_sceneflow: [B, 1, 375, 1242]
        predict_depth_l1_next = predict_depth_l1 + predict_sceneflow[:, 2:3, :, :]
        predict_disp_l1_next  = self.depth2disp_stereo(predict_depth_l1_next, intrinsic[:, 0, 0])
        target_depth_l1_next  = self.disp2depth_stereo(target_disp2_occ, intrinsic[:, 0, 0])
        disp_l2_metric, depth_l2_metric = self.eval_disp_depth(
            target_disp2_mask.bool(), target_disp2_occ, predict_disp_l1_next, target_depth_l1_next, predict_depth_l1_next)
        depth_l2_dict.update({key: depth_l2_metric[index] for index, key in enumerate(self.depth_metric)})

        scene_dict["d2"]                 = disp_l2_metric["otl"]
        disp_l2_otl_img                  = disp_l2_metric["otl_img"]
        outputs["otl2_img"]              = disp_l2_otl_img
        outputs["predict_disp_l1_next"]  = predict_disp_l1_next
        outputs["predict_depth_l1_next"] = predict_depth_l1_next


        ####################################################################################################
        ## SF
        ## Sceneflow 아웃라이어를 계산
        outlier_sceneflow = flow_outlier_epe.bool() + disp_l1_otl_img.bool() + disp_l2_otl_img.bool()
        outlier_sceneflow = outlier_sceneflow.float() * target_scene_mask
        scene_dict["sf"]  = (outlier_sceneflow.view(B, -1).sum(1)).mean() / 91873.4
        for key, value in scene_dict.items():
            if key in self.scene_metric:
                if value.device is "cpu":
                    scene_dict[key] = value.item()
                else:
                    scene_dict[key] = value.cpu().item()
        return scene_dict, depth_l1_dict, depth_l2_dict