import os
import sys
import warnings
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
from model_tool import *



class Trainer(object):
    def __init__(self, opt, device):
        warnings.filterwarnings("ignore")
        Tools().pytorch_randomness(random_seed = 1)
        self.opt    = opt
        if device is None:
            if torch.cuda.device_count() > 1:
                self.device = "cuda"
            else:
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        ## 규약
        ## Setting은 데이터로더, 모델, 옵티마이저, 로스, 각종 연산 클래스 등을 갖고 있음
        ## Computing은 모델 포워드를 구체적으로 정의하는 클래스
        ## Monitoring은 메트릭, 로스, 모델 저장 등 모델 모니터링이 가능
        self.setting = Setting(opt, self.device)
        self.compute = Computing(opt, self.device)
        self.control = Monitoring(opt, self.device)
    
    
    def process(self):
        print(">>> >>> >>> Training Start")
        total_loss_train = []
        total_loss_valid = []
        for epoch in range(self.opt.epoch):
            self.control.epoch = epoch
            ## 배치마다 로스를 누적할 리스트
            epoch_loss_train = []
            epoch_loss_valid = []
            ## 배치 depth 메트릭을 누적할 리스트
            eval_depth_train = []
            eval_depth_valid = []
            ## 배치 scene 메트릭을 누적할 리스트
            eval_scene_train = []
            eval_scene_valid = []
            

            self.setting.set_train()
            print(">>> Monocular dataset training")
            for batch_index, train_inputs in enumerate(tqdm(self.setting.train_mono_loader)):
                ## Forward MonoSceneFlow
                train_outputs, train_loss = self.batch_train(train_inputs)
                ## dp, sf, s2, s3, s3sm, total의 로스 딕셔너리
                epoch_loss_train.append(self.control.dict2list(train_loss))

                # ## 훈련 로스 프린트
                # self.control.loss_print(self.control.dict2list(train_loss), "Train Depth")

                ## [abs_rel, sqrt_rel, rmse, rmse_log, a1, a2, a3]의 딕셔너리
                ## train_metric 딕셔너리의 값을 리스트로 받음
                metric = self.batch_depth_eval(train_inputs, train_outputs)
                eval_depth_train.append(self.control.dict2list(metric))


            print(">>> Monocular dataset validation")
            for batch_index, valid_inputs in enumerate(tqdm(self.setting.valid_mono_loader)):
                ## Forward MonoSceneFlow
                valid_outputs, valid_loss = self.batch_valid(valid_inputs)
                ## dp, sf, s2, s3, s3sm, total의 로스 딕셔너리 (validation)
                epoch_loss_valid.append(self.control.dict2list(valid_loss))

                ## [abs_rel, sqrt_rel, rmse, rmse_log, a1, a2, a3]의 딕셔너리
                ## train_metric 딕셔너리의 값을 리스트로 받음
                metric = self.batch_depth_eval(valid_inputs, valid_outputs)
                eval_depth_valid.append(self.control.dict2list(metric))


            print(">>> Sceneflow dataset validation")
            for batch_index, scene_inputs in enumerate(tqdm(self.setting.valid_scene_loader)):
                ## Forward MonoSceneFlow dataset (200 images)
                ## SceneFlow 메트릭 [d1, f2, d2, sf] 딕셔너리를 받음
                scene, _, _ = self.batch_scene_eval(scene_inputs)
                ## sceneflow dataset의 sceneflow 메트릭과 depth 메트릭을 리스트로 바꾸어서 누적
                eval_scene_valid.append(self.control.dict2list(scene))




            self.setting.optim["scheduler"].step()
            ## 누적한 배치 로스, 메트릭들을 누적 axis wise로 평균 -> 한 에포크의 평균 로스 및 메트릭
            epoch_loss_train  = np.array(epoch_loss_train).mean(0)
            epoch_loss_valid  = np.array(epoch_loss_valid).mean(0)

            eval_depth_train  = np.array(eval_depth_train).mean(0)
            eval_depth_valid  = np.array(eval_depth_valid).mean(0)

            eval_scene_valid  = np.array(eval_scene_valid).mean(0)

            
            ## 로스와 메트릭 출력
            print("EPOCH   {0}".format(epoch+1))
            print(" ")
            self.control.loss_print(epoch_loss_train, "--- Train")
            self.control.depth_metric_print(eval_depth_train, "Train Depth")
            print(" ")
            self.control.loss_print(epoch_loss_valid, "--- Valid")
            self.control.depth_metric_print(eval_depth_valid, "Valid Depth")
            self.control.scene_metric_print(eval_scene_valid, "Valid Scene")
            print(" ")

            ## Model training infromation saving
            self.control.save(self.setting)



    def batch_train(self, inputs):
        ## Training에서는 Scene Augment를 사용
        with torch.no_grad():
            inputs = self.setting.augment_scene(inputs)
        outputs, loss = self.batch_process(inputs)

        self.setting.optim["optimizer"].zero_grad()
        loss["total"].backward()
        self.setting.optim["optimizer"].step()
        return outputs, loss



    def batch_valid(self, inputs):
        with torch.no_grad():
            ## Validation에서는 사이즈만 변경 (Only resize)
            inputs = self.setting.augment_resize(inputs)
            outputs, loss = self.batch_process(inputs)
        return outputs, loss



    def batch_process(self, inputs):
        outputs = {}
        # 토치 디바이스의 수가 1보다 큰 경우, 
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html 문서 참고
        # Please note that just calling my_tensor.to(device) returns a new copy of my_tensor on GPU 
        # instead of rewriting my_tensor. 
        # You need to assign it to a new tensor and use that tensor on the GPU.
        if torch.cuda.device_count() > 1:
            new_inputs            = self.compute.cpu2cuda(inputs)
            new_inputs, outputs   = self.compute.forward_model(new_inputs, outputs, self.setting)
            outputs, loss         = self.compute.compute_loss(new_inputs, outputs, self.setting)
            return outputs, loss
        else:
            inputs                = self.compute.cpu2cuda(inputs)
            inputs, outputs       = self.compute.forward_model(inputs, outputs, self.setting)
            outputs, loss         = self.compute.compute_loss(inputs, outputs, self.setting)
            return outputs, loss

    

    def batch_depth_eval(self, inputs, outputs):
        ## evaluation에서는 그래디언트 업데이트를 하지 않으므로 추적 X
        with torch.no_grad():
            metric = self.setting.loss["depth_metric"](inputs, outputs)
        return metric


    def batch_scene_eval(self, inputs):
        outputs = {}
        ## evaluation에서는 그래디언트 업데이트를 하지 않으므로 추적 X
        with torch.no_grad():
            ## Validation Augmentation (Only resize)
            inputs = self.compute.cpu2cuda(inputs)
            inputs = self.setting.augment_resize(inputs)
            inputs, outputs = self.compute.forward_model(inputs, outputs, self.setting)
            scene, _, _ = self.setting.loss["scene_metric"](inputs, outputs)
        return scene, _, _



if __name__ == "__main__":
    Trainer(options(), None).process()