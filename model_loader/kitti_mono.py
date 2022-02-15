import os
import random
import numpy as np
import skimage.io as io
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model_utility import *



def read_image(filename):
    return io.imread(filename)



def read_calib(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    # Read in a calibration file and parse into a dictionary.
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data



def calib2dict(path_dir):
    calibration_file_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
    intrinsic_dict_l = {}
    intrinsic_dict_r = {}
    for ii, date in enumerate(calibration_file_list):
        file_name = "cam_intrinsics/calib_cam_to_cam_" + date + '.txt'
        file_name_full = os.path.join(path_dir, file_name)
        file_data = read_calib(file_name_full)
        P_rect_02 = np.reshape(file_data['P_rect_02'], (3, 4))
        P_rect_03 = np.reshape(file_data['P_rect_03'], (3, 4))
        intrinsic_dict_l[date] = P_rect_02[:3, :3]
        intrinsic_dict_r[date] = P_rect_03[:3, :3]
    return intrinsic_dict_l, intrinsic_dict_r



def rescale_intrinsic(intrinsic_l, intrinsic_r, crop_info):
    str_x = crop_info[0]
    str_y = crop_info[1]
    intrinsic_l[0, 2] -= str_x
    intrinsic_l[1, 2] -= str_y
    intrinsic_r[0, 2] -= str_x
    intrinsic_r[1, 2] -= str_y
    return intrinsic_l, intrinsic_r



def crop_image(img_list, crop_info):    
    str_x = crop_info[0]
    str_y = crop_info[1]
    end_x = crop_info[2]
    end_y = crop_info[3]
    transformed = [img[str_y:end_y, str_x:end_x, :] for img in img_list]
    return transformed



class KITTIRawMonoDataset(Dataset):
    def __init__(self,
            datapath = None,
            use_flip = True,
            use_crop = True,
            crop_size = [370, 1224],
            num_examples = -1,
            index_file = None):
        view1 = 'image_02/data'
        view2 = 'image_03/data'
        point = "velodyne_points/data"
        ext   = '.jpg'
        self.datapath  = datapath
        self.seq_len   = 1
        self.use_flip  = use_flip
        self.use_crop  = use_crop
        self.crop_size = crop_size

        path_dir        = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        if not os.path.isdir(datapath):
            raise ValueError("Image directory '%s' not found!")

        filename_list   = [line.rstrip().split(' ') for line in index_file.readlines()]
        # filename_list   = filename_list[:int(len(filename_list) / 8)]
        self.image_list = []
        self.point_list = []
        
        for item in filename_list:
            date       = item[0][:10]
            scene      = item[0]
            point_date = scene.split("_drive")[0]
            idx_tgt    = item[1]
            idx_tgt_b  = '%.10d' % (int(idx_tgt) - 1)
            idx_tgt_f  = '%.10d' % (int(idx_tgt) + 1)

            # name_l0    = os.path.join(self.datapath, date, scene, view1, idx_tgt_b) + ext
            name_l1    = os.path.join(self.datapath, date, scene, view1, idx_tgt) + ext
            name_l2    = os.path.join(self.datapath, date, scene, view1, idx_tgt_f) + ext
            # name_r0    = os.path.join(self.datapath, date, scene, view2, idx_tgt_b) + ext
            name_r1    = os.path.join(self.datapath, date, scene, view2, idx_tgt) + ext
            name_r2    = os.path.join(self.datapath, date, scene, view2, idx_tgt_f) + ext
            point_1    = os.path.join(self.datapath, point_date, scene, point, idx_tgt) + ".bin"

            if os.path.isfile(name_l1) and os.path.isfile(name_l2) and \
               os.path.isfile(name_r1) and os.path.isfile(name_r2):
                self.image_list.append([name_l1, name_l2, name_r1, name_r2])
                self.point_list.append([point_1])
        if num_examples > 0:
            self.image_list = self.image_list[:num_examples]
        self.size = len(self.image_list)

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = calib2dict(path_dir)

        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(), transforms.transforms.ToTensor()])


    def load_point(self, point_path, cam):
        calib_path = os.path.join(self.datapath, point_path.split("/")[3].split("_drive")[0])
        ## 라이다 좌표계에서 L 카메라 좌표계로 변환
        ## cam == 2이면 L 카메라, cam == 3이면 R 카메라
        depth = ReadPoint().point2depth(calib_path, point_path, cam)
        depth = resize(depth, (375, 1242), order = 0, preserve_range = True, mode = "constant")
        ## 출력을 찍으면 0에서 80 사이의 값들
        depth = np.expand_dims(depth, axis = 0).astype(np.float32)
        return depth


    def __getitem__(self, index):
        index = index % self.size
        ## 이미지와 뎁스 로드
        image_list_np = [read_image(img) for img in self.image_list[index]]
        depth_list_np = [self.load_point(self.point_list[index][0], cam) for cam in [2, 3]]
        ## 배치 파일 이름의 정보, 카메라 행렬 로드
        image_l1_filename = self.image_list[index][0]
        basename          = os.path.basename(image_l1_filename)[:6]
        dirname           = os.path.dirname(image_l1_filename)[-51:]
        datename          = dirname[:10]
        intrinsic_l       = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        intrinsic_r       = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
        ## input size
        H, W, C    = image_list_np[0].shape
        input_size = torch.from_numpy(np.array([H, W])).float()

        ## use_crop이 True이면 이미지를 자름
        if self.use_crop:
            crop_h = self.crop_size[0]
            crop_w = self.crop_size[1]
            x = np.random.uniform(0, W - crop_w + 1)
            y = np.random.uniform(0, H - crop_h + 1)
            crop_info = [int(x), int(y), int(x + crop_w), int(y + crop_h)]
            ## 이미지와 카메라 행렬을 자른 것에 맞게 스케일링함
            image_list_np            = crop_image(image_list_np, crop_info)
            intrinsic_l, intrinsic_r = rescale_intrinsic(intrinsic_l, intrinsic_r, crop_info)
        
        ## 넘파이 어레이를 텐서로 변환
        image_list_tensor = [self.to_tensor(img) for img in image_list_np]
        depth_list_tensor = [torch.from_numpy(depth) for depth in depth_list_np]
        # image_l0    = image_list_tensor[0]
        image_l1    = image_list_tensor[0]
        image_l2    = image_list_tensor[1]
        # image_r0    = image_list_tensor[3]
        image_r1    = image_list_tensor[2]
        image_r2    = image_list_tensor[3]
        depth_l1    = depth_list_tensor[0]
        depth_r1    = depth_list_tensor[1]
        common_dict = {"index": index, "basename": basename, "datename": datename, "input_size": input_size}

        ## 좌 | 우 카메라 시퀀스를 뒤집으면 우 카메라가 좌로, 좌 카메라가 우로
        if self.use_flip is True and torch.rand(1) > 0.5:
            _, _, W = image_l1.size()
            image_l1_flip = torch.flip(image_l1, dims=[2])
            image_l2_flip = torch.flip(image_l2, dims=[2])
            image_r1_flip = torch.flip(image_r1, dims=[2])
            image_r2_flip = torch.flip(image_r2, dims=[2])
            depth_l1_flip = torch.flip(depth_l1, dims = [2])
            depth_r1_flip = torch.flip(depth_r1, dims = [2])
            intrinsic_l[0, 2] = W - intrinsic_l[0, 2]
            intrinsic_r[0, 2] = W - intrinsic_r[0, 2]

            example_dict = {
                "input_l1" : image_r1_flip,
                "input_l2" : image_r2_flip,
                "input_r1" : image_l1_flip,
                "input_r2" : image_l2_flip,   
                "depth_l1" : depth_r1_flip,
                "depth_r1" : depth_l1_flip,
                "kl": intrinsic_r,
                "kr": intrinsic_l}
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_l1" : image_l1,
                "input_l2" : image_l2,
                "input_r1" : image_r1,
                "input_r2" : image_r2,
                "depth_l1" : depth_l1,
                "depth_r1" : depth_r1,
                "kl": intrinsic_l,
                "kr": intrinsic_r,}
            example_dict.update(common_dict)
        return example_dict

    def __len__(self):
        return self.size



class KITTISplit_Train(KITTIRawMonoDataset):
    def __init__(self,
            datapath,
            use_flip=True,
            use_crop=True,
            crop_size=[370, 1224],
            num_examples=-1):
        super(KITTISplit_Train, self).__init__(
            datapath=datapath,
            use_flip=use_flip,
            use_crop=use_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_train.txt")



class KITTISplit_Valid(KITTIRawMonoDataset):
    def __init__(self,
            datapath,
            use_flip=False,
            use_crop=False,
            crop_size=[370, 1224],
            num_examples=-1):
        super(KITTISplit_Valid, self).__init__(
            datapath=datapath,
            use_flip=use_flip,
            use_crop=use_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_valid.txt")



class KITTISplit_Full(KITTIRawMonoDataset):
    def __init__(self,
            datapath,
            use_flip=True,
            use_crop=True,
            crop_size=[370, 1224],
            num_examples=-1):
        super(KITTISplit_Full, self).__init__(
            datapath=datapath,
            use_flip=use_flip,
            use_crop=use_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_full.txt")



class EigenSplit_Train(KITTIRawMonoDataset):
    def __init__(self,
            datapath,
            use_flip=True,
            use_crop=True,
            crop_size=[370, 1224],
            num_examples=-1):
        super(EigenSplit_Train, self).__init__(
            datapath=datapath,
            use_flip=use_flip,
            use_crop=use_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_train.txt")



class EigenSplit_Valid(KITTIRawMonoDataset):
    def __init__(self,
            datapath,
            use_flip=False,
            use_crop=False,
            crop_size=[370, 1224],
            num_examples=-1):
        super(EigenSplit_Valid, self).__init__(
            datapath=datapath,
            use_flip=use_flip,
            use_crop=use_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_valid.txt")



class EigenSplit_Full(KITTIRawMonoDataset):
    def __init__(self,
            datapath,
            use_flip=True,
            use_crop=True,
            crop_size=[370, 1224],
            num_examples=-1):
        super(EigenSplit_Full, self).__init__(
            datapath=datapath,
            use_flip=use_flip,
            use_crop=use_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_full.txt")



class EigenSplit_Test(Dataset):
    def __init__(self,
                 datapath,
                 num_examples=-1):
        view1 = 'image_02/data'
        view2 = 'image_03/data'
        point = "velodyne_points/data"
        ext   = '.jpg'
        self.datapath   = datapath
        index_file      = "index_txt/eigen_test.txt"
        path_directory  = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_directory, index_file)

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        if not os.path.isdir(datapath):
            raise ValueError("Image directory '%s' not found!", datapath)

        filename_list   = [line.rstrip().split(' ') for line in index_file.readlines()]
        self.image_list = []
        self.point_list = []
        for item in filename_list:
            # item: '2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000352.jpg'
            date       = item[0].split("/")[0]
            scene      = item[0].split("/")[1]
            # ./dataset/kitti/2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000032.jpg
            name_l1    = datapath + '/' + item[0]
            # 0000000512.jpg
            idx_tgt    = item[0].split('/')[4].split('.')[0]
            # 0000000513.jpg
            idx_src    = '%.10d' % (int(idx_tgt) + 1)
            name_l2    = name_l1.replace(idx_tgt, idx_src)
            point_1    = os.path.join(datapath, date, scene, point, idx_tgt[:10]) + ".bin"
            if not os.path.isfile(name_l2):
                idx_prev = '%.10d' % (int(idx_tgt) - 1)
                name_l2  = name_l1.replace(idx_tgt, idx_prev)          

            self.image_list.append([name_l1, name_l2])
            self.point_list.append([point_1])

        if num_examples > 0:
            self.image_list = self.image_list[:num_examples]

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = calib2dict(path_directory)
        self.to_tensor = transforms.Compose(
            [transforms.ToPILImage(), transforms.transforms.ToTensor()])

        self.size = len(self.image_list)


    def load_point(self, point_path, cam):
        calib_path = os.path.join(self.datapath, point_path.split("/")[3].split("_drive")[0])
        ## 라이다 좌표계에서 L 카메라 좌표계로 변환
        ## cam == 2이면 L 카메라, cam == 3이면 R 카메라
        depth = ReadPoint().point2depth(calib_path, point_path, cam)
        depth = resize(depth, (375, 1242), order = 0, preserve_range = True, mode = "constant")
        ## 출력을 찍으면 0에서 80 사이의 값들
        depth = np.expand_dims(depth, axis = 0).astype(np.float32)
        return depth


    def __getitem__(self, index):
        index = index % self.size
        im_l1_filename = self.image_list[index][0]
        im_l2_filename = self.image_list[index][1]
        pt_l1_filename = self.point_list[index][0]

        ## read images and flow
        im_l1_np = read_image(im_l1_filename)
        im_l2_np = read_image(im_l2_filename)
        pt_l1_np = self.load_point(pt_l1_filename, 2)
    
        ## example filename
        basename = os.path.dirname(im_l1_filename).split('/')[-3] + '_' + os.path.basename(im_l1_filename).split('.')[0]
        dirname  = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]

        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
        im_l1 = self.to_tensor(im_l1_np)
        im_l2 = self.to_tensor(im_l2_np)
        dp_l1 = torch.from_numpy(pt_l1_np)

        # input size
        H, W, C       = im_l1_np.shape
        input_im_size = torch.from_numpy(np.array([H, W])).float()
    
        example_dict = {
            "input_l1": im_l1,
            "input_l2": im_l2,
            "depth_l1": dp_l1,
            "index": index,
            "basename": basename,
            "datename": datename,
            "kl": k_l1,
            "input_size": input_im_size}
        return example_dict


    def __len__(self):
        return self.size