from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .correlation_package.correlation import Correlation



## 모델 가중치 초기화
def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LeakyReLU):
            pass
        elif isinstance(layer, nn.Sequential):
            pass



class MonoSceneFlow(nn.Module):
    def __init__(self, opt):
        super(MonoSceneFlow, self).__init__()
        self.evaluation   = True
        self.num_chs      = [3, 32, 64, 96, 128, 192, 256]
        self.output_level = 4
        self.search_range = 4
        self.num_levels   = 7
        self.dim_corr     = (self.search_range * 2 + 1) ** 2 # 코스트 볼륨 차원은 81로 제한
        self.corr_params  = {
            "pad_size": self.search_range, 
            "max_disp": self.search_range, 
            "corr_multiply": 1,
            "kernel_size": 1, 
            "stride1": 1, 
            "stride2": 1}
        ## 네트워크 정의
        self.feature_encoder  = FeatureExtractor(self.num_chs)
        ## context network는 입력 채널이 32 + 3 + 1 = 36인 입력 채널을 받아서 context 정보 추출
        ## multimono-sf에서는 context network를 사용하지 않음
        self.context_networks = ContextNetwork(32 + 3 + 1)
        self.scene_warping    = WarpingSceneFlow()
        ## 활성함수 설정
        self.leaky_relu       = nn.LeakyReLU(0.1, inplace = True)
        self.sigmoid          = nn.Sigmoid()

        ## 레이어를 담을 모듈리스트 설정
        self.upconv_layers = nn.ModuleList()
        self.flow_decoders = nn.ModuleList()
        for index, cha in enumerate(self.num_chs[::-1]):
            if index > self.output_level:
                break
            if index == 0: 
                # 코스트 볼륨하고 피처맵을 concat할 것, index == 0이면 코스트 볼륨 채널 + 피처맵 채널
                num_ch_in = self.dim_corr + cha # 81 + ch
            else: 
                # l이 0이 아니고 self.output_level보다 작으면 upconvolution을 하기 위한 컨볼루션 레이어 선언
                # 코스트 볼륨 채널 + 피처맵 채널 + conv_out 채널 + scene 채널 + disp 채널
                num_ch_in = self.dim_corr + cha + 32 + 3 + 1 
                self.upconv_layers.append(upconv(num_in_layers = 32, num_out_layers = 32, kernel_size = 3, scale = 2))
            self.flow_decoders.append(MonoSceneFlowDecoder(ch_in = num_ch_in))
        initialize_msra(self.modules())



    def pwc_net(self, input_dict, x1_raw, x2_raw, intrinsic):
        ## 이미지 인코더, 두 이미지를 sharing encoder에 포워드
        output_dict = {}
        x1_pyramid  = self.feature_encoder(x1_raw) + [x1_raw]
        x2_pyramid  = self.feature_encoder(x2_raw) + [x2_raw]
        ## outputs
        disps_1      = []
        disps_2      = []
        sceneflows_f = []
        sceneflows_b = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            if l == 0: ## 가장 작은 피처맵은 x1_warp = x1, x2_warp = x2로 초기값을 줌
                x1_warp = x1
                x2_warp = x2
            else:
                ## L 레이어의 x_out, scene, disp의 사이즈를 L-1 레이어의 x1 사이즈로 upsampling 하는 코드
                x1_out  = self.upconv_layers[l-1](x1_out)
                x2_out  = self.upconv_layers[l-1](x2_out)
                scene_f = interpolate_as(scene_f, x1, mode = "bilinear")
                scene_b = interpolate_as(scene_b, x1, mode = "bilinear")
                disp_l1 = interpolate_as(disp_l1, x1, mode = "bilinear") ## scene_f와 같은 프레임에 대응
                disp_l2 = interpolate_as(disp_l2, x1, mode = "bilinear") ## scene_b와 같은 프레임에 대응
                ## disp_l1과 scene_f (l1 -> l2)를 이용해서 warp_grid를 만들고 x2를 x1의 뷰로 와핑
                ## disp_l2와 scene_b (l2 -> l1)를 이용해서 warp_grid를 만들고 x1를 x2의 뷰로 와핑
                x2_warp = self.scene_warping(x2, disp_l1, scene_f, intrinsic, input_dict['aug_size'])
                x1_warp = self.scene_warping(x1, disp_l2, scene_b, intrinsic, input_dict['aug_size'])


            ## 레이어 노말리제이션
            ## if self.opt.use_normalize == True:
            ##     x1, x1_warp, x2, x2_warp = self.normalize_features([x1, x1_warp, x2, x2_warp])
            ## elif self.opt.use_normalize == False:
            ##     pass
            ## 어떤 index이든 피처와 와핑 피처 간의 코스트 볼륨을 계산
            ## [B, C, H, W] * [B, C, H, W] -> [B, 81, H, W]으로 계산됨
            out_corr_relu_f = self.leaky_relu(Correlation.apply(x1, x2_warp, self.corr_params))
            out_corr_relu_b = self.leaky_relu(Correlation.apply(x2, x1_warp, self.corr_params))

            ## MonoSceneFlowEsitmator (MonoSceneFlowDecoder)
            if l == 0:
                x1_out, scene_f, disp_l1 = self.flow_decoders[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, scene_b, disp_l2 = self.flow_decoders[l](torch.cat([out_corr_relu_b, x2], dim=1))
            else:
                x1_out, scene_f_res, disp_l1 = self.flow_decoders[l](torch.cat([out_corr_relu_f, x1, x1_out, scene_f, disp_l1], dim=1))
                x2_out, scene_b_res, disp_l2 = self.flow_decoders[l](torch.cat([out_corr_relu_b, x2, x2_out, scene_b, disp_l2], dim=1))
                scene_f = scene_f + scene_f_res
                scene_b = scene_b + scene_b_res

            ## https://github.com/visinf/multi-mono-sf/blob/main/models/model_monosceneflow.py#L192
            ## multimono-sf에서는 context_network가 문제가 있다고 판단하고, 이를 제거하였음
            ## index가 output_level에 도달하면 warping upsampling을 break
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                sceneflows_f.append(scene_f)
                sceneflows_b.append(scene_b)                
            else:
                scene_res_f, disp_l1 = self.context_networks(torch.cat([x1_out, scene_f, disp_l1], dim=1))
                scene_res_b, disp_l2 = self.context_networks(torch.cat([x2_out, scene_b, disp_l2], dim=1))
                scene_f = scene_f + scene_res_f
                scene_b = scene_b + scene_res_b
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)        
                sceneflows_f.append(scene_f)
                sceneflows_b.append(scene_b)        
                break

        x1_rev = x1_pyramid[::-1]
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['scene_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['scene_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        return output_dict


    def forward(self, input_dict):
        output_dict = {}
        ## 좌측 이미지 포워드 (모노큘라 비디오)
        output_dict = self.pwc_net(
            input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['kl_aug'])
        

        if self.training: ## Right -> flip -> forward -> output flip
            ## 스테레오 매칭을 통해 디스패리티 로스 계산
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            kr_flip_aug   = input_dict["kr_flip_aug"]
            output_dict_r = self.pwc_net(input_dict, input_r1_flip, input_r2_flip, kr_flip_aug)
            for index in range(0, len(output_dict_r['scene_f'])):
                output_dict_r['scene_f'][index] = flow_horizontal_flip(output_dict_r['scene_f'][index])
                output_dict_r['scene_b'][index] = flow_horizontal_flip(output_dict_r['scene_b'][index])
                output_dict_r['disp_l1'][index] = torch.flip(output_dict_r['disp_l1'][index], [3])
                output_dict_r['disp_l2'][index] = torch.flip(output_dict_r['disp_l2'][index], [3])
            output_dict['output_dict_r'] = output_dict_r


        if self.evaluation: ## Post Processing 
            ## self.evaluation True이면 실행 -> sceneflow metric에서 post processing 연산을 사용할 예정
            ## eval 단계에서는 좌측 이미지와 K를 flip하여 포워드 함
            input_l1_flip    = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip    = torch.flip(input_dict['input_l2_aug'], [3])
            kl_flip_aug      = input_dict["kl_flip_aug"]
            output_dict_flip = self.pwc_net(input_dict, input_l1_flip, input_l2_flip, kl_flip_aug)

            scene_f_pp = []
            scene_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            for index in range(0, len(output_dict_flip['scene_f'])):
                scene_f_pp.append(post_processing(output_dict['scene_f'][index], flow_horizontal_flip(output_dict_flip['scene_f'][index])))
                scene_b_pp.append(post_processing(output_dict['scene_b'][index], flow_horizontal_flip(output_dict_flip['scene_b'][index])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][index], torch.flip(output_dict_flip['disp_l1'][index], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][index], torch.flip(output_dict_flip['disp_l2'][index], [3])))
            output_dict['scene_f_pp'] = scene_f_pp
            output_dict['scene_b_pp'] = scene_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
        return output_dict