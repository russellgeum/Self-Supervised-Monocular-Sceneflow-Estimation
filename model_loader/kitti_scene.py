import os
import png
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model_utility import *



VALIDATE_INDICES = [
    2,   34,  35,  36,  37, 38,  39,  40,  41,  42, 77,  78,  79,  80,  81, 
    83,  99, 100, 101, 102, 105, 106, 112, 113, 114, 115, 116, 117, 133, 141, 
    144, 145, 167, 187, 190, 191, 192, 193, 195, 199]


WIDT_TO_DATE = dict()
WIDT_TO_DATE[1242] = '2011_09_26'
WIDT_TO_DATE[1224] = '2011_09_28'
WIDT_TO_DATE[1238] = '2011_09_29'
WIDT_TO_DATE[1226] = '2011_09_30'
WIDT_TO_DATE[1241] = '2011_10_03'



def load_jpg_image(filename):
    return io.imread(filename)



def load_png_flow(flow_file):
    flow_object = png.Reader(filename = flow_file)
    flow_direct = flow_object.asDirect()
    flow_data   = list(flow_direct[2])
    (w, h)      = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow[:, :, 0:2], (1 - invalid_idx * 1)[:, :, None]



def load_png_disp(disp_file):
    disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
    disp_np = np.expand_dims(disp_np, axis=2)
    mask_disp = (disp_np > 0).astype(np.float64)
    return disp_np, mask_disp



def numpy2torch(array):
    # 넘파이 타입을 텐서 타입으로 변환, 이미지 외의 flow, disp에 적용
    assert(isinstance(array, np.ndarray))
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array.copy()).float()



def scaling_intrinsic(intrinsic_l, intrinsic_r, crop_info):
    """
    intrinsic matrix 스케일링
    원본 사이즈에 매칭하는 intrinsic matrix를 크롭 사이즈에 맞게 rescaling
    """
    str_x  = crop_info[0]
    str_y  = crop_info[1]
    intrinsic_l[0, 2] -= str_x 
    intrinsic_l[1, 2] -= str_y 
    intrinsic_r[0, 2] -= str_x 
    intrinsic_r[1, 2] -= str_y 
    return intrinsic_l, intrinsic_r


def cropping_images(image_list, crop_info):
    str_x = crop_info[0]
    str_y = crop_info[1]
    end_x = crop_info[2]
    end_y = crop_info[3]
    cropped = [image[str_y:end_y, str_x:end_x, :] for image in image_list]
    return cropped


class KITTISceneFlow(Dataset):
    def __init__(self, datapath = None, use_resize = True, dstype = "full", ext = ".jpg"):
        """
        KITTI 2015 Scene Flow Dataset 구성
        - 200 장의 이미지로 구성
        - 좌/우 이미지 4장
        - 좌/우 occ disp 2장
        - 좌/우 noc disp 2장
        - 좌/우 이미지간의 occ flow 1장
        - 좌/우 이미지간의 noc flow 1장
        - 그 외 obj_map, viz_occ, calib_root
        scene flow 메트릭 계산에 필수인 요소
        datapath: "./dataset/kitti" ... 
        use_resize: 사전에 정의한 크기 (375, 1242)로 resize 여부 옵션
        dstype: "train", "valid", "full" 중 선택 가능
        """
        self.datapath   = datapath
        self.use_resize = use_resize
        self.dstype     = dstype
        self.img_ext    = ext
        num_images      = 200
        images_l_root   = os.path.join(datapath, "data_scene_flow", "training", "image_2")
        images_r_root   = os.path.join(datapath, "data_scene_flow", "training", "image_3")
        flow_occ_root   = os.path.join(datapath, "data_scene_flow", "training", "flow_occ")
        flow_noc_root   = os.path.join(datapath, "data_scene_flow", "training", "flow_noc")
        disp0_occ_root  = os.path.join(datapath, "data_scene_flow", "training", "disp_occ_0")
        disp1_occ_root  = os.path.join(datapath, "data_scene_flow", "training", "disp_occ_1")
        disp0_noc_root  = os.path.join(datapath, "data_scene_flow", "training", "disp_noc_0")
        disp1_noc_root  = os.path.join(datapath, "data_scene_flow", "training", "disp_noc_1")

        ## loading image -------------------------------------------------------------------
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory {} not found!".format(images_l_root))
        if not os.path.isdir(images_r_root):
            raise ValueError("Image directory {} not found!".format(images_r_root))
        if not os.path.isdir(flow_occ_root):
            raise ValueError("Image directory {} not found!".format(flow_occ_root))
        if not os.path.isdir(flow_noc_root):
            raise ValueError("Image directory {} not found!".format(flow_noc_root))
        if not os.path.isdir(disp0_occ_root):
            raise ValueError("disparity directory {} not found!".format(disp0_occ_root))
        if not os.path.isdir(disp1_occ_root):
            raise ValueError("disparity directory {} not found!".format(disp1_occ_root))
        if not os.path.isdir(disp0_noc_root):
            raise ValueError("disparity directory {} not found!".format(disp0_noc_root))
        if not os.path.isdir(disp1_noc_root):
            raise ValueError("disparity directory {} not found!".format(disp1_noc_root))

        ## train, val, full에 따라 mono sceneflow 데이터셋을 분할
        validate_indices = [x for x in VALIDATE_INDICES if x in range(num_images)]
        if dstype == "train":
            list_of_indices = [x for x in range(num_images) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(num_images)
        else:
            raise ValueError("KITTI: dstype {} unknown!".format(dstype))
        
        ## 선택한 데이터셋의 모든 경로를 구성해서 리스트로 구성
        self.image_list = []
        self.flow_list  = []
        self.disp_list  = []
        # list_of_indices = list_of_indices[:5]
        for index in list_of_indices:
            file_idx  = '%.6d' % index
            
            image_l1  = os.path.join(images_l_root, file_idx + "_10" + self.img_ext)
            image_l2  = os.path.join(images_l_root, file_idx + "_11" + self.img_ext)
            image_r1  = os.path.join(images_r_root, file_idx + "_10" + self.img_ext)
            image_r2  = os.path.join(images_r_root, file_idx + "_11" + self.img_ext)
            flow_occ  = os.path.join(flow_occ_root, file_idx + "_10.png")
            flow_noc  = os.path.join(flow_noc_root, file_idx + "_10.png")
            disp0_occ = os.path.join(disp0_occ_root, file_idx + "_10.png")
            disp1_occ = os.path.join(disp1_occ_root, file_idx + "_10.png")
            disp0_noc = os.path.join(disp0_noc_root, file_idx + "_10.png")
            disp1_noc = os.path.join(disp1_noc_root, file_idx + "_10.png")
            ## 해당 파일 경로가 없으면 오류 발생
            for _, item in enumerate(
                [image_l1, image_l2, image_r1, image_r2, flow_occ, flow_noc, disp0_occ, disp1_occ, disp0_noc, disp1_noc]):
                if not os.path.isfile(item):
                    raise ValueError("File not exist: %s", item)

            self.image_list.append([image_l1, image_l2, image_r1, image_r2])
            self.flow_list.append([flow_occ, flow_noc])
            self.disp_list.append([disp0_occ, disp1_occ, disp0_noc, disp1_noc])

        self.size = len(self.image_list)
        assert len(self.image_list) != 0

        # Intrinsic matrix 로드
        self.intrinsic_dict_l    = {}
        self.intrinsic_dict_r    = {}
        self.intrinsic_dict_l, self.intrinsic_dict_r = ReadImage().read_all_cam2cam(datapath)
        print(">>> Sceneflow dataset image length is {}".format(len(self.image_list)))
        print(">>> Sceneflow dataset flow length  is {}".format(len(self.flow_list)))
        print(">>> Sceneflow dataset disp length  is {}".format(len(self.disp_list)))
        print(">>> Sceneflow dataset Loaded intrinsic matrix!")

        # transformation np2torch, 이미지를 불러온 경우에, 노말라이즈까지 적용
        self.to_tensor = transforms.transforms.ToTensor()


    def __getitem__(self, index):
        index = index % self.size
        ## 이미지 파일 로드 // 플로우 파일 로드 // 디스패리티 파일 로드
        image_list  = [load_jpg_image(filename) for filename in self.image_list[index]]
        flow_list   = [load_png_flow(filename) for filename in self.flow_list[index]]
        flow_list   = [img for sub_list in flow_list for img in sub_list]
        disp_list   = [load_png_disp(path) for path in self.disp_list[index]]
        disp_list   = [img for sub_list in disp_list for img in sub_list]

        ## intrnisic_l, intrinsic_r를 로드 후 토치 텐서로 변환
        basename    = os.path.basename(self.image_list[index][0][:6]) # outputs에 포함한 파일 이름
        intrinsic_l = torch.from_numpy(self.intrinsic_dict_l[WIDT_TO_DATE[image_list[0].shape[1]]]).float()
        intrinsic_r = torch.from_numpy(self.intrinsic_dict_r[WIDT_TO_DATE[image_list[2].shape[1]]]).float()

        ## 원본 이미지의 사이즈를 input_image_size 이름으로 텐서 생성
        H, W, C = np.array(image_list[0]).shape
        input_image_size = torch.from_numpy(np.array([H, W])).float()

        if self.use_resize:
            crop_h, crop_w = 370, 1224
            x = np.random.uniform(0, W - crop_w + 1)
            y = np.random.uniform(0, H - crop_h + 1)
            crop_info = [int(x), int(y), int(x+crop_w), int(y+crop_h)]

            image_list = cropping_images(image_list, crop_info)
            flow_list  = cropping_images(flow_list, crop_info)
            disp_list  = cropping_images(disp_list, crop_info)
            intrinsic_l, intrinsic_r = scaling_intrinsic(intrinsic_l, intrinsic_r, crop_info)


        img_list_tensor  = [self.to_tensor(img) for img in image_list]
        flow_list_tensor = [numpy2torch(img) for img in flow_list]
        disp_list_tensor = [numpy2torch(img) for img in disp_list]
        
            
        """
        향후 필요하다면 이미지 스케일링에 따라 intrinsic matrix로 스케일링하고
        이를 input_l1_aug, ... K_l_aug, K_r_aug ... 형태로 딕셔너리에 전달
        """
        output_dict = {
            "input_l1": img_list_tensor[0],
            "input_l2": img_list_tensor[1],
            "input_r1": img_list_tensor[2],
            "input_r2": img_list_tensor[3],
            "kl": intrinsic_l,
            "kr": intrinsic_r,

            "target_flow": flow_list_tensor[0],
            "target_flow_mask": flow_list_tensor[1],
            "target_flow_noc": flow_list_tensor[2],
            "target_flow_mask_noc": flow_list_tensor[3],

            "target_disp1": disp_list_tensor[0],
            "target_disp1_mask": disp_list_tensor[1],
            "target_disp2_occ": disp_list_tensor[2],
            "target_disp2_mask_occ": disp_list_tensor[3],

            "target_disp1_noc": disp_list_tensor[4],
            "target_disp1_mask_noc": disp_list_tensor[5],
            "target_disp2_noc": disp_list_tensor[6],
            "target_disp2_mask_noc": disp_list_tensor[7],

            "index": index,
            "basename": basename,
            "input_size": input_image_size}
        return output_dict


    def __len__(self):
        return self.size




class KITTISceneFlow_Test(Dataset):
    def __init__(self, datapath):
        images_l_root = os.path.join(datapath, "data_scene_flow", "testing", "image_2_jpg")
        images_r_root = os.path.join(datapath, "data_scene_flow", "testing", "image_3_jpg")
        
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory %s not found!", images_l_root)
        if not os.path.isdir(images_r_root):
            raise ValueError("Image directory %s not found!", images_r_root)

        num_images = 200
        list_of_indices = range(num_images)

        path_dir = os.path.dirname(os.path.realpath(__file__))
        self.image_list = []
        self.flow_list = []
        self.disp_list = []
        img_ext = '.jpg'

        for ii in list_of_indices:

            file_idx = '%.6d' % ii

            im_l1 = os.path.join(images_l_root, file_idx + "_10" + img_ext)
            im_l2 = os.path.join(images_l_root, file_idx + "_11" + img_ext)
            im_r1 = os.path.join(images_r_root, file_idx + "_10" + img_ext)
            im_r2 = os.path.join(images_r_root, file_idx + "_11" + img_ext)
           

            file_list = [im_l1, im_l2, im_r1, im_r2]
            for _, item in enumerate(file_list):
                if not os.path.isfile(item):
                    raise ValueError("File not exist: %s", item)

            self.image_list.append([im_l1, im_l2, im_r1, im_r2])

        self.size = len(self.image_list)
        assert len(self.image_list) != 0

        ## loading calibration matrix
        # Intrinsic matrix 로드
        self.intrinsic_dict_l    = {}
        self.intrinsic_dict_r    = {}
        self.intrinsic_dict_l, self.intrinsic_dict_r = ReadImage().read_all_cam2cam(datapath)
        print(">>> Sceneflow dataset image length is {}".format(len(self.image_list)))
        print(">>> Sceneflow dataset flow length  is {}".format(len(self.flow_list)))
        print(">>> Sceneflow dataset disp length  is {}".format(len(self.disp_list)))
        print(">>> Sceneflow dataset Loaded intrinsic matrix!")

        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(), transforms.transforms.ToTensor()])


    def __getitem__(self, index):
        index = index % self.size
        im_l1_filename = self.image_list[index][0]
        im_l2_filename = self.image_list[index][1]
        im_r1_filename = self.image_list[index][2]
        im_r2_filename = self.image_list[index][3]
        ## read float32 images and flow
        im_l1_np = load_jpg_image(im_l1_filename)
        im_l2_np = load_jpg_image(im_l2_filename)
        im_r1_np = load_jpg_image(im_r1_filename)
        im_r2_np = load_jpg_image(im_r2_filename)
        
        ## example filename
        basename = os.path.basename(im_l1_filename)[:6]

        ## find intrinsic
        k_l1  = torch.from_numpy(self.intrinsic_dict_l[WIDT_TO_DATE(im_l1_np.shape[1])]).float()
        k_r1  = torch.from_numpy(self.intrinsic_dict_r[WIDT_TO_DATE(im_r1_np.shape[1])]).float()
        im_l1 = self.to_tensor(im_l1_np)
        im_l2 = self.to_tensor(im_l2_np)
        im_r1 = self.to_tensor(im_r1_np)
        im_r2 = self.to_tensor(im_r2_np)

        ## input size
        H, W, _       = im_l1_np.shape
        input_im_size = torch.from_numpy(np.array([H, W])).float()

        example_dict = {
            "input_l1": im_l1,
            "input_l2": im_l2,
            "input_r1": im_r1,
            "input_r2": im_r2,
            "kl": k_l1,
            "kr": k_r1,
            "index": index,
            "basename": basename,
            "input_size": input_im_size}
        return example_dict


    def __len__(self):
        return self.size