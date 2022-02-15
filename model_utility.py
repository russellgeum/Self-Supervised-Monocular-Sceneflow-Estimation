import os
import re
import sys
import time
import json
import random
from tqdm import tqdm
from scipy import misc
import math
import numpy as np
from collections import Counter

import png
import cv2
import open3d as open3d
from PIL import Image
import skimage.io as io
import skimage.transform
from skimage.color import rgb2gray
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
KITTI 데이터셋을 위한 함수 모듈
class SplitData
    데이터셋을 나누기 위한 정적 메서드
class ReadImage
    이미지 데이터을 읽기 위한 정적 메서드
class ReadPoint
    포인트 클라우드 데이터를 읽기 위한 정적 메서드
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Data(object):
    def __init__(self):
        pass
        """
        KITTI data의 split을 준비하는 함수
        eigen split의 규격을 따른다.
        """
    @staticmethod
    def readlines(datapath):
        # Read all the lines in a text file and return as a list
        with open(datapath, 'r') as f:
            lines = f.read().splitlines()
        return lines


    @staticmethod
    def savelines(filename, datapath: str):
        f = open(datapath, "w")
        for data in filename:
            f.write(data + "\n") 
        f.close()


    @staticmethod
    def removelines(datapath, filename, frame_ids):
        """
        for KITTI
        Args:
            filename:  ['2011_09_26/2011_09_26_drive_0057_sync 311 l', 
                        '2011_09_26/2011_09_26_drive_0035_sync 130 r]
            frame_ids: [-3, -2, -1, 0, 1, 2]
        """
        modified_key = []
        side_map     = {"2": 2, "3": 3, "l": 2, "r": 3}
        for _, data in enumerate(filename):
            line = data.split()
            name = line[0]      # 2011_09_26/2011_09_26_drive_0035_sync or
            key  = int(line[1]) # 311 or 130
            side = line[2]      # l or r
            length = len(os.listdir(
                datapath + "/" + name + "/" + "image_0{}/data".format(side_map[side])))
            
             # 포함되어 있다면
            if key in list(range(-frame_ids[0], length - frame_ids[-1] - 1)):
                modified_key.append(data) # 수정된 키 리스트에 추가
            else:
                pass
        return modified_key



# Image, Scene, Disparity 등의 데이터를 로드하는 함수 (중요)
class ReadImage(object):
    def __init__(self):
        pass


    @staticmethod
    def read_PFM(file):
        import re
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale


    @staticmethod
    def read_flo_file(filename):
        """
        https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593

        Read from Middlebury .flo file
        :param flow_file: name of the flow file
        :return: optical flow data in matrix
        """
        f = open(filename, 'rb')
        magic = np.fromfile(f, np.float32, count=1)
        data2d = None

        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            #print("Reading %d x %d flow file in .flo format" % (h, w))
            flow = np.ones((h[0],w[0],3))
            data2d = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
            # reshape data into 3D array (columns, rows, channels)
            data2d = np.resize(data2d, (h[0], w[0], 2))
            flow[:,:,:2] = data2d
        f.close()
        return flow


    @staticmethod
    def read_pfm_file(flow_file):
        """
        https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593

        Read from .pfm file
        :param flow_file: name of the flow file
        :return: optical flow data in matrix
        """
        (data, scale) = ReadImage().readPFM(flow_file)
        return data 


    @staticmethod
    def read_png_file(flow_file):
        """
        https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593

        Read from KITTI .png file
        :param flow_file: name of the flow file
        :return: optical flow data in matrix
        """
        # https://github.com/visinf/self-mono-sf/blob/054df08674a2df96885682e657ca4803c129a364/datasets/common.py#L67
        flow_object = png.Reader(filename = flow_file)
        flow_direct = flow_object.asDirect()
        flow_data   = list(flow_direct[2])
        (w, h)      = flow_direct[3]['size']
        flow        = np.zeros((h, w, 3), dtype=np.float64)
        for i in range(len(flow_data)):
            flow[i, :, 0] = flow_data[i][0::3]
            flow[i, :, 1] = flow_data[i][1::3]
            flow[i, :, 2] = flow_data[i][2::3]

        invalid_idx = (flow[:, :, 2] == 0)
        flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
        flow[invalid_idx, 0] = 0
        flow[invalid_idx, 1] = 0
        return flow


    @staticmethod
    def read_calib_file(path):
        """
        https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593

        Read in a calibration file and parse into a dictionary.
        """
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data


    @staticmethod
    def read_image(path):
        # https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593
        image = np.array(Image.open(path))
        return image
    

    @staticmethod
    def read_disp(path):
        # https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593
        if ".png" in path:
            disp = Image.open(path)
            disp = np.ascontiguousarray(disp, dtype = np.float32) / 256.
        else:
            ReadImage().read_PFM(path)[0]

        # # https://github.com/visinf/self-mono-sf/blob/054df08674a2df96885682e657ca4803c129a364/datasets/common.py#L85
        # 0 ~ 1
        # disp = cv2.imread(path).astype(np.uint16) / 256.0
        # disp = np.expand_dims(disp, axis=2)
        disp = skimage.transform.resize(
            disp, (1242, 375)[::-1], order = 0, preserve_range = True, mode = "constant")
        disp = np.reshape(disp, (disp.shape[0], disp.shape[1], 1))
        return disp


    @staticmethod
    def read_flow(path):
        """
        https://github.com/gengshan-y/rigidmask/blob/a6532a8a83076462820232f27a1bafb2ec519037/utils/flowlib.py#L593
        read optical flow data from flow file
        :param filename: name of the flow file
        :return: optical flow data in numpy array
        """
        if path.endswith('.flo'):
            flow = ReadImage().read_flo_file(path)
        elif path.endswith('.pfm'):
            flow = ReadImage().read_pfm_file(path)
        elif path.endswith('.png'):
            flow = ReadImage().read_png_file(path)
        else:
            raise Exception('Invalid flow file format!')
        flow = skimage.transform.resize(
            flow, (1242, 375)[::-1], order = 0, preserve_range = True, mode = "constant")
        return flow[:, :, 0:2]


    @staticmethod
    def read_all_cam2cam(datapath): # 모든 날짜의 cam2cam 파일을 불러오는 함수
        calibration_file_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        intrinsic_dict_l      = {}
        intrinsic_dict_r      = {}

        for _, date in enumerate(calibration_file_list):
            file_name = datapath + "/" + date + "/calib_cam_to_cam.txt"
            file_data = ReadImage().read_calib_file(file_name)
            P_rect_02 = np.reshape(file_data['P_rect_02'], (3, 4))
            P_rect_03 = np.reshape(file_data['P_rect_03'], (3, 4))
            intrinsic_dict_l[date] = P_rect_02[:3, :3]
            intrinsic_dict_r[date] = P_rect_03[:3, :3]
        return intrinsic_dict_l, intrinsic_dict_r


    @staticmethod
    def read_cam2cam(path):
        """
        ArgsL
            path: cam2cam 파일 경로
        return
            L/R intrinsic matrix: 4x4 행렬
        """
        file_data = ReadImage().read_calib_file(path)

        # P_rect_02 파일의 L/R intrinsic matrix 로드
        P_rect_02   = np.reshape(file_data['P_rect_02'], (3, 4))
        P_rect_03   = np.reshape(file_data['P_rect_03'], (3, 4))
        intrinsic_l = P_rect_02[:3, :3]
        intrinsic_r = P_rect_03[:3, :3]
        
        identity_l         = np.eye(4)
        identity_r         = np.eye(4)
        identity_l[:3, :3] = intrinsic_l
        identity_r[:3, :3] = intrinsic_r
        identity_l         = identity_l.astype(np.float32)
        identity_r         = identity_r.astype(np.float32)
        return identity_l, identity_r



## 포인트 클라우드를 읽어들이는 함수 (중요)
class ReadPoint(object):
    def __init__(self):
        pass
        """
        포인트 클라우드를 불러오는 메서드들을 모아놓은 것
        @staticmethod
        def read_velo2cam
        def read_velodyne_points
        def sub2ind
        def point2depth
        """
    @staticmethod
    def read_velo2cam(path):
        """
        벨로다인 캘리브레이션 파일을 읽어들이는 함수
        Read KITTI calibration file
        (from https://github.com/hunse/kitti)
        """
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass
        return data


    @staticmethod
    def read_velodyne_points(filename):
        """
        Load 3D point cloud from KITTI file format
        (adapted from https://github.com/hunse/kitti)
        """
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous
        return points


    @staticmethod
    def sub2ind(matrixSize, rowSub, colSub):
        """
        Convert row, col matrix subscripts to linear indices
        """
        m, n = matrixSize
        return rowSub * (n-1) + colSub - 1


    @staticmethod
    def point2depth(calib_path, point_path, cam = 2, vel_depth = False):
        """
        캘리브레이션 경로와 벨로다인 파일 경로를 읽어서 뎁스 맵을 만드는 함수
        Args:
            calib_path: ./dataset/kitti-master/2011_09_26
            point_path: ./dataset/kitti-master/2011_09_26/2011_09_26_drive_0022_sync/velodyne_points/data/0000000473.bin

        returns:
            GT depth image (np.max: 80.0, np.min: 0.1)
            shape: [375, 1242]
        """
        # 1. load calibration files
        cam2cam  = ReadPoint().read_velo2cam(
            os.path.join(calib_path + "/" + "calib_cam_to_cam.txt"))
        velo2cam = ReadPoint().read_velo2cam(
            os.path.join(calib_path + "/" + "calib_velo_to_cam.txt"))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # 2. 이미지 모양을 획득 (375, 1242)
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
        
        # 3.
        # 3차원 포인트 점을 카메라 좌표계로 변환하고 다시 K를 곱해서 이미지로 사영시키는 수식
        # 먼저 4x4 항등행렬을 선언하고 여기서 3x3 부분은 회전 행렬을 붙인다. (R_rect_00)
        # 그리고 모션 벡터를 cam2cam의 P_rect_0 성분을 불러와서 둘을 np.dot한다.
        # 마지막으로 velo2cam 매트릭스를 np.dot하면 벨로다인 포인트 -> 이미지로 사영하는 매트릭스를 만듬
        R_cam2rect = np.eye(4)                                  # 4x4 항등행렬
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3) # 회전 운동
        P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)     # 모션 벡터
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        # 4.
        # 벨로다인 포인트 클라우드를 불러오고, x, y, z, 1의 homogenous 좌표계로 만듬
        # load velodyne points and remove all behind image plane (approximation)
        # each row of the velodyne data is forward, left, up, reflectance
        velo = ReadPoint().read_velodyne_points(point_path)
        velo = velo[velo[:, 0] >= 0, :]

        # 5. 벨로다인 포인트 homogenous 값을 카메라의 이미지 좌표에 사영하는 계산과정 이미지 = 사영행렬 * 3차원 벨로다인 포인트
        velo_pts_im        = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] # shape is (포인트 갯수, x, y, 1 값)

        if vel_depth:
            velo_pts_im[:, 2] = velo[:, 0]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        # 1. velo_path_im.shape는 3개 (x, y, 1) 성분이 61021개 있다. 여기의 x, y 좌표에서 1씩 빼준 것을 다시 velo_pts_im[:, 0] and [:, 1]에 대입
        # 2. 그리고 x 좌표가 0 이상이고 y 좌표가 0 이상인 값만 유효한 인덱스로 취급한다.
        # 3. 그리고 val_ind 이면서 동시에 velo_pts_im 좌표의 위치가 이미지의 크기보다 작은 것만 다시 val_inds로 할당 (그래야만 이미지에 좌표가 잘 맺히므로)
        # 4. 마지막으로 그 유효한 좌표의 위치, 즉 True만 velo_pts_im로 취급
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds          = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds          = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
        velo_pts_im       = velo_pts_im[val_inds, :]

        depth = np.zeros((im_shape[:2])) # 이미지로 사영, 375, 1245 사이즈의 zero map을 만듬
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # 마지막, 중복된 값을 제거
        inds = ReadPoint().sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts   = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

        depth[depth < 0] = 0
        return depth






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
옵티컬 플로우, Scene Flow 시각화를 위한 도구 모음
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
UNKNOWN_FLOW_THRESH = 1e7
class Visualization():
    def __init__(self):
        pass


    ## 시각화 하지 않는 disparity를 png 파일로 저장하는 함수
    @staticmethod
    def wrtie_disp(filename, disp):
        io.imsave(filename, (disp * 256.0).astype(np.uint16))


    ## 시각화 하지 않는 optical flow를 png 파일로 저장하는 함수
    @staticmethod
    def write_flow(filename, flow, v = None, mask = None):
        if v is None:
            assert (flow.ndim == 3)
            assert (flow.shape[2] == 2)
            u = flow[:, :, 0]
            v = flow[:, :, 1]
        else:
            u = flow
        assert (u.shape == v.shape)

        height_img, width_img = u.shape
        if mask is None:
            valid_mask = np.ones([height_img, width_img])
        else:
            valid_mask = mask

        flow_u = np.clip((u * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
        flow_v = np.clip((v * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
        output = np.stack((flow_u, flow_v, valid_mask), axis=-1)
        with open(filename, 'wb') as f:
            writer = png.Writer(width = width_img, height = height_img, bitdepth=16)
            writer.write(f, np.reshape(output, (-1, width_img*3)))


    @staticmethod
    ## 시각화 하지 않는 disparity png 파일을 읽는 함수
    def read_disp(disp_file):
        disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
        disp_np = np.expand_dims(disp_np, axis=2)
        mask_disp = (disp_np > 0).astype(np.float64)
        return disp_np


    @staticmethod
    ## 시각화 하지 않는 optical flow png 파일을 읽는 함수
    def read_flow(flow_file):
        flow = cv2.imread(flow_file, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)[:,:,::-1].astype(np.float64)
        flow, valid = flow[:, :, :2], flow[:, :, 2:]
        flow = (flow - 2**15) / 64.0
        return flow




    ###################################################################################################
    ## 여기서부터는 아예 시각화된 flow, depth를 저장하는 함수
    ###################################################################################################
    @staticmethod
    def write_vis_disp(filename, disp):
        plt.imsave(filename, disp, cmap = 'magma', vmax = np.percentile(disp, 95))


    @staticmethod
    def write_vis_flow(filename, flow):
        ## 시각화를 하는 optical flow, disp 데이터 저장
        io.imsave(filename, flow)


    @staticmethod
    def make_color():
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
        col += + BM

        # MR
        colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255
        return colorwheel


    @staticmethod
    def compute_color(u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [H, W] = u.shape
        img    = np.zeros([H, W, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx]  = 0
        v[nanIdx]  = 0
        colorwheel = Visualization().make_color()
        ncols      = np.size(colorwheel, 0)

        rad = np.sqrt(u ** 2 + v ** 2)
        a   = np.arctan2(-v, -u) / np.pi
        fk  = (a + 1) / 2 * (ncols - 1) + 1

        k0  = np.floor(fk).astype(int)
        k1  = k0 + 1
        k1[k1 == ncols + 1] = 1
        f   = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp  = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col  = (1 - f) * col0 + f * col1

            idx      = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx   = np.logical_not(idx)

            col[notidx]  *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
        return img


    ## optcial flow (u, v)에 색깔을 입히는 함수
    @staticmethod
    def flow2png(flow):
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return:     optical flow image in middlebury color
        """
        flow = flow.transpose([1, 2, 0])
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad)) 
        # maxrad = 4

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = Visualization().compute_color(u, v)
        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0
        return np.uint8(img)


    @staticmethod
    def show_image(image, option = "torch", size = (10, 4), cmap = "magma", show_disp = True):
        """
        토치나 텐서플로우 형태의 이미지를 받아서 이미지를 플롯하는 함수
        Args: type
            Pytorch:    [C, H, W]
            Tensorflow: [H, W, C]
            numpy:      [H, W, C]
        """
        plt.rcParams["figure.figsize"] = size
        ## 이미지가 np.ndarray 타입이면 바로 플랏
        if isinstance(image, np.ndarray):
            ## shape이 [C, H, W]이거나 [3, H, W] 이면, [H, W, C] 형태로 transpose
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            ## shape이 [1, C, H, W]이면 [C, H, W]로 줄이고 [H, W, C]로 transpose
            elif len(image.shape) == 4:
                image = np.squeeze(image, axis = 0)
                image = np.transpose(image, (1, 2, 0))
        else:
            ## 이미지의 gradient 추적이 True이면 detach()
            if image.requires_grad == True:
                image = image.detach()
            ## 이미지의 devcie가 gpu이면 cpu()로 바꿈
            if image.device is not "cpu":
                image = image.cpu().numpy()
            else:
                image = image.numpy()

            if option == "torch":
                ## shape이 [C, H, W]이거나 [3, H, W] 이면, [H, W, C] 형태로 transpose
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
                ## shape이 [1, C, H, W]이면 [C, H, W]로 줄이고 [H, W, C]로 transpose
                elif len(image.shape) == 4:
                    image = np.squeeze(image, axis = 0)
                    image = np.transpose(image, (1, 2, 0))

        """
        uint8 -> float32로 바꾸면 엄청 큰 정수 단위의 float32가 됨
        따리서 255.로 나누어 주는 것이 중요
        그리고 cv2.imread로 불러온 이미지를 plt.imshow로 띄울때는 cv2.COLOR_BGR2RGB
        """
        if np.min(image) < 0:
            image = (image * 255).astype(np.uint8)

        if show_disp:
            plt.imshow(image, cmap = cmap, vmax = np.percentile(image, 95))
            plt.axis('off')
            plt.show()
        else:
            plt.imshow(image, cmap = cmap)
            plt.axis('off')
            plt.show()
        




## 배치데이터 샘플링, 이미지 플랏 등 기타 사용 함수
class Tools(object):
    def __init__(self):
        pass


    @staticmethod
    def batch_cpu2cuda(batch, device):
        for key in batch:
            if key not in ["basename", "index", "datename"]:
                batch[key] = batch[key].to(device)
        return batch
        

    @staticmethod
    def sample_batch(loader, end): # 모델 데이터로터에서 배치 샘플 하나를 추출
        test = []
        start = time.time()
        for index, data in tqdm(enumerate(loader)):
            test.append(data)
            if end == index:
                break
            elif end == "all":
                pass
        print("batch sampling time:  ", time.time() - start)
        return test


    @staticmethod
    def show_graph(data, xlabel, ylabel, title, color, marker, linestyle):
        """
        data:   np.array
        xlabel: str
        ylabel: str
        title:  str
        color:  str
        marker: "o" or 
        """
        plt.rcParams["figure.figsize"] = (4, 4)
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = True 
        plt.rc('font', size = 10)        # 기본 폰트 크기
        plt.rc('axes', labelsize = 15)   # x,y축 label 폰트 크기
        plt.rc('figure', titlesize = 15) # figure title 폰트 크기

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot(data, c = color, marker = marker, linestyle = linestyle)
        plt.show()

    
    @staticmethod
    def show_hist(data_list, range = None, bins = None):
        """
        data: [data1, datat2, data3 ... ...]
        """
        plt.rcParams["figure.figsize"] = (10, 8)
        for data in data_list:
            plt.hist(data, range = None, bins = None)
        plt.show()



    @staticmethod
    ## pytorch randomnetss
    def pytorch_randomness(random_seed):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(random_seed)
        np.random.seed(random_seed+1)
        torch.manual_seed(random_seed+2)
        torch.cuda.manual_seed(random_seed+3)
        # torch.cuda.manual_seed_all(4) # if use multi-GPU