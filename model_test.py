import os
import sys
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model_utility import *
from model_option import *
from model_loader import *
from model_layer import *
from model_loss import *
from model_save import *



def load_weight(weight_path):
    """
    official code와 reproduce code 간의 스위칭
    모델의 키 값이 안 맞으면 키 값을 바꾸어 weight를 로드하는 함수
    """
    old_weight   = torch.load(os.path.join("./model_save", weight_path))
    new_weight = {}
    if "state_dict" in old_weight: ## .ckpt 파일인 경우
        for key in old_weight["state_dict"]:
            original_key = key
            key   = key.split(".")
            index = key.index("_model")
            del key[index]
            key   = ".".join(key)
            if "feature_pyramid_extractor" in key.split("."):
                key = key.split(".")
                index = key.index("feature_pyramid_extractor")
                key[index] = "feature_encoder"
                key = ".".join(key)
                new_weight[key] = old_weight["state_dict"][original_key]
            elif "flow_estimators" in key.split("."): 
                key = key.split(".")
                index = key.index("flow_estimators")
                key[index] = "flow_decoders"
                key = ".".join(key)
                new_weight[key] = old_weight["state_dict"][original_key]
            else:
                new_weight[key] = old_weight["state_dict"][original_key]
    else:
        new_weight.update(old_weight)
    return new_weight



def load_loader(datapath = "./dataset/kitti", use_resize = False, dstype = "full"):
    ## 필요한 dataset 인스턴스와 augmentation 인스턴스 생성
    ## 모노큘라 키티 데이터 로더
    mono_dataset   = EigenSplit_Test(datapath)
    ## SceneFlow 키티 데이터 로더
    scene_dataset  = KITTISceneFlow(datapath, use_resize = use_resize, dstype = dstype)
    
    mono_loader    = DataLoader(dataset = mono_dataset, batch_size = 1, shuffle = False)
    scene_loader   = DataLoader(dataset = scene_dataset, batch_size = 1, shuffle = False)
    return mono_loader, scene_loader



def load_model(device = "cuda:0", weight = "model80.pt"):
    model = MonoSceneFlow(None).to(device)
    model.load_state_dict(load_weight(weight))
    model.eval()
    return model



def eval_depth(model, loader, device):
    augment_resize = AugmentResize([256, 832]).to(device)
    depth_eval     = MonoDepthEvaluation().to(device)

    with torch.no_grad():
        depth_metrics = []
        image_list    = []
        depth_list    = []

        for batch_idx, inputs in enumerate(tqdm(loader)):
            for key in inputs:
                if key not in ["index", "basename", "datename"]:
                    inputs[key] = inputs[key].to(device)
            ## 모델 포워드
            inputs       = augment_resize(inputs)
            outputs      = model(inputs)
            ## 모노뎁스 evaluation
            mono_metric  = depth_eval(inputs, outputs)

            depth_metrics.append([mono_metric[key] for key in mono_metric])
            image_list.append(inputs["input_l1_aug"][0].cpu().numpy())
            depth_list.append(outputs["disp_l1_pp"][0][0].cpu().numpy())

        depth_metrics = np.array(depth_metrics).mean(0)
        
    print("Mono Depth Metric")
    for index, key in enumerate(["abs_rel", "sqrt_rel", "rmse", "rmse_log", "a1", "a2", "a3"]):
        print("{}  {:0.3f}  ".format(key, depth_metrics[index]), end = " ")
    print(" ")
    return image_list, depth_list



def eval_scene(model, loader, flow_name, disp1_name, disp2_name, device, use_save = False):
    augment_resize = AugmentResize([256, 832]).to(device)
    scene_eval     = MonoSceneFlowEvaluation().to(device)

    ## Sceneflow 시각화를 위해 저장할 데이터 경로
    flow_foldername  = os.path.join("./model_save", flow_name)
    disp1_foldername = os.path.join("./model_save", disp1_name)
    disp2_foldername = os.path.join("./model_save", disp2_name)
    ## 해당 폴더들이 없으면, 폴더 생성
    if use_save == True:
        if not os.path.exists(flow_foldername):
            os.makedirs(flow_foldername)
            print("created save folder {}".format(flow_foldername))
        if not os.path.exists(disp1_foldername):
            os.makedirs(disp1_foldername)
            print("created save folder {}".format(disp1_foldername))
        if not os.path.exists(disp2_foldername):
            os.makedirs(disp2_foldername)
            print("created save folder {}".format(disp2_foldername))
    else:
        print("use save is False, Not create save folder")


    with torch.no_grad():
        scene_metric  = []
        image_list    = []
        flow_list     = []
        disp_list     = []

        for batch_idx, inputs in enumerate(tqdm(loader)):
            for key in inputs:
                if key not in ["index", "basename", "datename"]:
                    inputs[key] = inputs[key].to(device)
            # 모델 포워드
            inputs    = augment_resize(inputs)
            outputs   = model(inputs)
            
            ## scene evaluation 계산
            loss_dict, _, _ = scene_eval(inputs, outputs)
            scene_metric.append([loss_dict[key] for key in loss_dict])

            ## SceneFlow 시각화에 필요한 이미지, optical flow, disp1, disp2 (== disp_l1_next)
            image_l1             = inputs["input_l1_aug"][0].cpu().numpy()
            optical_flow         = outputs["optical_flow"].cpu().numpy()
            predict_disp_l1      = outputs["predict_disp_l1"].cpu().numpy()
            predict_disp_l1_next = outputs["predict_disp_l1_next"].cpu().numpy()

            ## 데이터를 시각화 표현으로 변환
            vis_flow  = Visualization().flow2png(optical_flow[0, ...])
            vis_disp1 = predict_disp_l1[0, 0, ...]
            vis_disp1 = (vis_disp1 / vis_disp1.max() * 255).astype(np.uint8)
            vis_disp2 = predict_disp_l1_next[0, 0, ...]
            vis_disp2 = (vis_disp2 / vis_disp2.max() * 255).astype(np.uint8)

            image_list.append(image_l1)
            flow_list.append(vis_flow)
            disp_list.append(vis_disp1)
                

            ## 데이터들을 저장한다면 use_save == True
            if use_save == True:
                file_index = inputs["index"][0].cpu().numpy()
                ## 시각화 하지 않는 optical flow, disp 저장 경로 (png 타입)
                not_vis_flow_name  = os.path.join(flow_foldername, "%.6d_10.png" % int(file_index))
                not_vis_disp1_name = os.path.join(disp1_foldername, "%.6d_10.png" % int(file_index))
                not_vis_disp2_name = os.path.join(disp2_foldername, "%.6d_11.png" % int(file_index))
                ## 시각화 하지 않는 optical flow, disp 데이터 저장
                Visualization().write_flow(not_vis_flow_name, optical_flow[0, ...].swapaxes(0, 1).swapaxes(1, 2))
                Visualization().wrtie_disp(not_vis_disp1_name, predict_disp_l1[0, 0, ...])
                Visualization().wrtie_disp(not_vis_disp2_name, predict_disp_l1_next[0, 0, ...])
                
                ## 시각화를 하는 optical flow, disp 저장 경로 (옵티컬 플로우는 png 타입, disp는 jpg 타입)
                vis_flow_name  = os.path.join(flow_foldername, "vis_%.6d_10.png" % int(file_index))
                vis_disp1_name = os.path.join(disp1_foldername, "vis_%.6d_10.jpg" % int(file_index))
                vis_disp2_name = os.path.join(disp2_foldername, "vis_%.6d_11.jpg" % int(file_index))
                ## 시각화를 하는 optical flow, disp 데이터 저장
                Visualization().write_vis_flow(vis_flow_name, vis_flow)
                Visualization().write_vis_disp(vis_disp1_name, vis_disp1)
                Visualization().write_vis_disp(vis_disp2_name, vis_disp2)
            else:
                pass


    ## 배치마다의 메트릭으로 axis = 0, 배치 수로 평균
    scene_metric = np.array(scene_metric).mean(0)
    for index, key in zip([0, 1, 2, 3], ["d1", "f1", "d2", "sf"]):
        print("{} {:0.4f}".format(key, scene_metric[index]), end = " ")
    print(" ")
    return image_list, flow_list, disp_list




if __name__ == "__main__":
    mono_loader, scene_loader = load_loader()
    model = load_model()