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



class Setting(object):
    def __init__(self, opt, device):
        self.opt    = opt
        self.device = device
        # if opt.pose_frames == "all":
        #     self.num_pose_frames = len(opt.frame_ids)
        # else:
        #     self.num_pose_frames = 2
        print(">>> Device:       {}".format(self.device))
        print(">>> Device count: {}".format(torch.cuda.device_count()))
        print(">>> Epoch         {}".format(opt.epoch))
        print(">>> Batch         {}".format(opt.batch))
        print(">>> LR            {}".format(opt.learning_rate))
        print(">>> Img size      ({}, {})".format(opt.height, opt.width))
        print(" ")


        ## 데이터 파일 로드 (eigen split or kitti split) for monodepth2
        # self.dataset   = opt.datapath.split("/")[2]
        # self.filepath  = opt.splits + "/" + opt.datatype + "/{}_files.txt"
        # train_filename = Data().readlines(self.filepath.format("train"))
        # valid_filename = Data().readlines(self.filepath.format("val"))
        # 경험상 데이터의 1/8 정도로 하면 학습이 되는 경향성을 확인할 수 있음
        # train_filename = train_filename[ :len(train_filename) // 8]
        # valid_filename = valid_filename[ :len(valid_filename) // 8]
        # 경험상 40개의 데이터셋은 학습은 되지 않음, 메트릭 스케일이 적절한지만 확인
        # train_filename = train_filename[ :40]
        # valid_filename = valid_filename[ :40]

        
        ## 데이터로더 설정
        self.train_mono_loader  = self.set_mono_loader(self.opt.splits, "train", True, self.opt.batch)
        self.valid_mono_loader  = self.set_mono_loader(self.opt.splits, "valid", False, 1)

        self.valid_scene_loader = self.set_scene_lodaer(dstype = "valid")
        print(">>> ALL DATASET IS LOADED...")
        print(">>> KITTIMonoDataset(Train), KITTIMonoDataset(Valid), KITTISceneDataset(Valid)")
        print(" ")
        print(">>> DATASET INFORMATION")
        print(">>> MonoDataset splits                    {}".format(self.opt.splits))
        print(">>> MonoDataset, Num of train batch       {}".format(self.train_mono_loader.__len__()))
        print(">>> MonoDataset, Num of valid batch       {}".format(self.valid_mono_loader.__len__()))
        print(">>> MonoDataset, Num of total train batch {}".format(opt.epoch * self.train_mono_loader.__len__()))
        print(">>> MonoDataset, Num of total valid batch {}".format(opt.epoch * self.train_mono_loader.__len__()))
        print(">>> SceneDataset Num of valid batch       {}".format(self.valid_scene_loader.__len__()))
        print(">>> Setting DataLoader")
        print(" ")


        ## 데이터 augmentation을 위한 인스턴스 생성
        self.augment_scene  = AugmentSceneFlow(
            self.opt.batch, self.device, True, 0.07, [0.93, 1.00], [self.opt.height, self.opt.width])
        self.augment_scene  = self.augment_scene.to(self.device)
        self.augment_resize = AugmentResize([self.opt.height, self.opt.width])
        self.augment_resize = self.augment_resize.to(self.device)

        self.model = {}
        self.parameters = []
        self.set_model()

        self.loss = {}
        self.set_loss()

        self.optim = {}
        self.set_optim()
        

    ## 키티 데이터 로더를 설정하는 함수
    def set_mono_loader(self, splits = "eigen", types = "train", shuffle = True, batch = 4):
        if splits == "kitti":
            if types == "train":
                dataset = KITTISplit_Train(self.opt.datapath, True, True)
            elif types == "valid":
                dataset = KITTISplit_Valid(self.opt.datapath, False, False)
        elif splits == "eigen":
            if types == "train":
                dataset = EigenSplit_Train(self.opt.datapath, True, True)
            elif types == "valid":
                dataset = EigenSplit_Valid(self.opt.datapath, False, False)
                
        dataloader = DataLoader(dataset, batch, shuffle, num_workers = self.opt.num_workers, drop_last = True)
        print(">>> Dataset type {} option {}".format(splits, types))
        if shuffle == True:
            print(">>> Dataset train length  {}".format(dataset.size))
        elif shuffle == False:
            print(">>> Dataset valid length  {}".format(dataset.size))
        return dataloader

    
    def set_scene_lodaer(self, dstype = "valid"):
        ## Validation 용도이면 use_resize == False
        dataset    = KITTISceneFlow(self.opt.datapath, False, dstype, ".jpg")
        dataloader = DataLoader(dataset, 1, False, num_workers = self.opt.num_workers)
        return dataloader


    def set_model(self):
        ## Mono Depth + Sceneflow 네트워크 로드
        self.model["model"] = MonoSceneFlow(opt = self.opt)
        ## 모델의 파라미터를 로드
        for key in self.model:
            self.model[key] = self.model[key].to(self.device)
            self.parameters += list(self.model[key].parameters())
        print(">>> Setting model")


    def set_loss(self):
        ## Scene + Depth Loss and Depth, Scene Eval
        ## 오클루션 마스크를 사용하지 않는 로스
        self.loss["loss"]         = SelfSceneFlowLoss()
        # self.loss["loss"]         = SelfSceneFlowLoss_NotPoint()
        self.loss["depth_metric"] = MonoDepthEvaluation()
        self.loss["scene_metric"] = MonoSceneFlowEvaluation()
        self.loss.update({key: self.loss[key].to(self.device) for key in self.loss})
        print(">>> Setting loss")

    
    def set_optim(self):
        self.optim["optimizer"] = torch.optim.Adam(self.parameters, self.opt.learning_rate)
        self.optim["scheduler"] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer = self.optim["optimizer"], milestones = [23, 39, 47, 54], gamma = 0.5)
        print(">>> Setting optim")

    
    def set_train(self):
        for value in self.model.values():
            value.train()


    def set_valid(self):
        for value in self.model.values():
            value.eval()