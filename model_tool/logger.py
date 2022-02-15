import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model_utility import *
from model_loader import *
from model_layer import *
from model_loss import *



class Monitoring(object):
    def __init__(self, opt, device):
        """
        control
            def print
            def save
            def load
            def metric
        """
        self.opt       = opt
        self.device    = device
        self.epoch     = 0
        self.loss_name    = ["dp", "sf", "s2", "s3", "s3sm", "total"]
        # self.loss_name    = ["dp", "sf", "s2", "s3sm", "total"]
        self.depth_metric = ["abs_rel", "sqrt_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        self.scene_metric = ["d1", "f1", "d2", "sf"]


    def dict2list(self, metric_dict: dict):
        """
        Args:
            metric_dict (dict): ["abs_rel", "sqrt_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
            와 같은 하나의 태스크에 대한 메트릭들이 값과 함께 딕셔너리 형태
        Return:
            metric_list (list): metric_dict 딕셔너리의 items에서 value만 뽑아 리스트로 만듬

        예시 1)
        ["abs_rel", "sqrt_rel", "rmse", "rmse_log", "a1", "a2", "a3"] 뎁스 메트릭을 items()로 순회해서 리스트에 추가
        예시 2)
        ["d1", "f1", "d2", "sf"] 플로우 메트릭을 items()로 순회해서 리스트에 추가
        """
        metric_list = [value for key, value in metric_dict.items()]
        return metric_list


    
    def loss_print(self, loss_list: list, title: str):
        print(title, end = " ")
        print(" ")
        for index, key in enumerate(self.loss_name):
            print("{}  {:0.4f}  |  ".format(key, loss_list[index]), end = " ")
        print(" ")

        

    def depth_metric_print(self, depth_metric: list, title: str):
        print(title, end = " ")
        print(" ")
        for index, key in enumerate(self.depth_metric):
            print("{}  {:0.4f}  |  ".format(key, depth_metric[index]), end = " ")
        print(" ")



    def scene_metric_print(self, scene_metric: list, title: str):
        print(title, end = " ")
        print(" ")
        for index, key in enumerate(self.scene_metric):
            print("{}  {:0.4f}  |  ".format(key, scene_metric[index]), end = " ")
        print(" ")
        


    def save(self, setting):
        save_directory = os.path.join("./model_save", self.opt.save)
        loss_directory = os.path.join(save_directory, "loss")
        
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        if not os.path.isdir(loss_directory):
            os.makedirs(loss_directory)
        
        # 주기적으로 특정 epoch마다 모델 정보를 저장
        if (self.epoch+1) % self.opt.freq == 0:
            for key in setting.model:
                torch.save(setting.model[key].state_dict(),
                    os.path.join(save_directory, key + str(self.epoch+1) + ".pt"))

        # 가장 마지막 epoch의 모델 정보를 저장
        if (self.epoch+1) == self.opt.epoch:
            for key in setting.model:
                torch.save(setting.model[key].state_dict(),
                    os.path.join(save_directory, key + str(self.epoch+1) + ".pt"))

            # for key in self.metric_name: # 모델의 로그 기록 저장
            #     np.save(os.path.join(loss_directory, key + ".npy"), train_log[key])
            #     np.save(os.path.join(loss_directory, key + ".npy"), valid_log[key])
    

    @staticmethod
    def load(model, weight):
        raise NotImplementedError