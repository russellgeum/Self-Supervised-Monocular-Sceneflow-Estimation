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



class Computing(object):
    def __init__(self, opt, device):
        """
        compute
            def forward
            def pose
            def depth
            def loss
        """
        self.opt    = opt
        self.device = device
        # if opt.pose_frames == "all":
        #     self.num_pose_frames = len(opt.frame_ids)
        # else:
        #     self.num_pose_frames = 2


    def cpu2cuda(self, inputs):
        if torch.cuda.device_count() > 1:
            new_inputs = {}
            for key in inputs:
                if key not in ["index", "basename", "datename"]:
                    new_inputs[key] = inputs[key].to(self.device)
            return new_inputs
        elif torch.cuda.device_count() == 1:
            for key in inputs:
                if key not in ["index", "basename", "datename"]:
                    inputs[key] = inputs[key].to(self.device)
            return inputs


    def forward_model(self, inputs, outputs, setting):
        outputs = setting.model["model"](inputs)
        return inputs, outputs


    def compute_loss(self, inputs, outputs, setting):
        loss = setting.loss["loss"](inputs, outputs)
        return outputs, loss