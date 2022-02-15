import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_layer import *



# intrinsic matrix rescaling
def intrinsic_scale(intrinsic, sx, sy):
    out = intrinsic.clone()
    out[:, 0, 0] *= sx
    out[:, 0, 2] *= sx
    out[:, 1, 1] *= sy
    out[:, 1, 2] *= sy
    return out



# intrinsic matrix crop
def intrinsic_crop(intrinsic, str_x, str_y):    
    out = intrinsic.clone()
    out[:, 0, 2] -= str_x
    out[:, 1, 2] -= str_y
    return out




class PhotometricAugmentation(nn.Module):
    def __init__(self):
        super(PhotometricAugmentation, self).__init__()
        self.min_gamma  = 0.8
        self.max_gamma  = 1.2
        self.min_brght  = 0.5
        self.max_brght  = 2.0
        self.min_shift  = 0.8
        self.max_shift  = 1.2
        self.intv_gamma = self.max_gamma - self.min_gamma
        self.intv_brght = self.max_brght - self.min_brght
        self.intv_shift = self.max_shift - self.min_shift


    def forward(self, *args):
        _, C, _, _  = args[0].size()
        num_splits  = len(args)
        concat_data = torch.cat(args, dim = 1)
        d_dtype     = concat_data.dtype
        d_device    = concat_data.device
        b, c, h, w  = concat_data.size()
        num_images  = int(c / C)

        rand_gamma  = torch.rand([b, 1, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self.intv_gamma + self.min_gamma
        rand_bright = torch.rand([b, 1, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self.intv_brght + self.min_brght
        rand_shift  = torch.rand([b, 3, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self.intv_shift + self.min_shift

        # gamma
        concat_data = concat_data ** rand_gamma.expand(-1, c, h, w)

        # brightness
        concat_data = concat_data * rand_bright.expand(-1, c, h, w)

        # color shift
        rand_shift  = rand_shift.expand(-1, -1, h, w)
        rand_shift  = torch.cat([rand_shift for i in range(0, num_images)], dim=1)
        concat_data = concat_data * rand_shift

        # clip
        concat_data = torch.clamp(concat_data, 0, 1)
        split = torch.chunk(concat_data, num_splits, dim=1)
        return split




class IdentityParameters(nn.Module):
    def __init__(self):
        super(IdentityParameters, self).__init__()
        self.batch_size = 0
        self.device     = None
        self.o = None
        self.i = None
        self.identity_params = None


    def update(self, batch_size, device):
        self.o  = torch.zeros([batch_size, 1, 1], device = device).float()
        self.i  = torch.ones([batch_size, 1, 1], device = device).float()
        r1      = torch.cat([self.i, self.o, self.o], dim=2)
        r2      = torch.cat([self.o, self.i, self.o], dim=2)
        r3      = torch.cat([self.o, self.o, self.i], dim=2)
        return torch.cat([r1, r2, r3], dim=1)


    def forward(self, batch_size, device):
        if self.batch_size != batch_size or self.device != device:
            self.identity_params = self.update(batch_size, device)
            self.batch_size      = batch_size
            self.device          = device
        return self.identity_params.clone()




class AugmenScaleCrop(nn.Module):
    def __init__(self, 
                batch, 
                device, 
                photometric = True, 
                trans = 0.07, 
                scale = [0.93, 1.0], 
                resize = [256, 832]):
        super(AugmenScaleCrop, self).__init__()
        self.photometric = photometric
        # Augmentation Parameters
        self.max_trans   = trans
        self.min_scale   = scale[0]
        self.max_scale   = scale[1]
        self.resize      = resize
        self.batch       = batch
        self.device      = device
        self.meshgrid    = Meshgrid()
        self.photo_aug   = PhotometricAugmentation()
        self.identity    = IdentityParameters()


    # scale, rot, tx, ty 값을 dim = 1로 합침
    def compose_params(self, scale, rot, tx, ty):
        return torch.cat([scale, rot, tx, ty], dim=1)


    # dim = 1에서 compose_params를 scale, rot, tx, ty로 분리함
    def decompose_params(self, params):
        return params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]


    def find_invalid(self, img_size, params): # 이 코드가 무엇을 하는 코드인지 잘 모르겠음
        scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image
        intm_size_h = torch.floor(img_size[0] * scale)
        intm_size_w = torch.floor(img_size[1] * scale)

        ## 4 representative points of the intermediate images
        hf_h = (intm_size_h - 1.0) / 2.0
        hf_w = (intm_size_w - 1.0) / 2.0        
        hf_h.unsqueeze_(1)
        hf_w.unsqueeze_(1)
        hf_o    = torch.zeros_like(hf_h)
        hf_i    = torch.ones_like(hf_h)
        pt_mat  = torch.cat([torch.cat([hf_w, hf_o, hf_o], dim=2), torch.cat([hf_o, hf_h, hf_o], dim=2), torch.cat([hf_o, hf_o, hf_i], dim=2)], dim=1)
        ref_mat = torch.ones(self.batch, 4, 3, device=self.device)
        ref_mat[:, 1, 1] = -1
        ref_mat[:, 2, 0] = -1
        ref_mat[:, 3, 0] = -1
        ref_mat[:, 3, 1] = -1
        ref_pts = torch.matmul(ref_mat, pt_mat).transpose(1,2)

        ## Perform trainsform
        tform_mat = self.identity(self.batch, self.device)
        tform_mat[:, 0, 2] = tx[:, 0]
        tform_mat[:, 1, 2] = ty[:, 0]   
        pts_tform = torch.matmul(tform_mat, ref_pts)

        ## Check validity: whether the 4 representative points are inside of the original images
        img_hf_h = (img_size[0] - 1.0) / 2.0
        img_hf_w = (img_size[1] - 1.0) / 2.0
        x_tf     = pts_tform[:, 0, :]
        y_tf     = pts_tform[:, 1, :]

        invalid = (((x_tf <= -img_hf_w) | (y_tf <= -img_hf_h) | (x_tf >= img_hf_w) | (y_tf >= img_hf_h)).sum(dim=1, keepdim=True) > 0).float()
        return invalid


    def find_aug_params(self, img_size, resize): # 이 코드가 무엇을 하는 코드인지 잘 모르겠음
        # scale: for the size of intermediate images (original * scale = intermediate image)
        # rot and trans: rotating and translating of the intermedinate imageㄴㅇㄴㅇㄴㅇㄴㅇㄴㅇ
        # then resize the augmented images into the resize image

        # 배치 사이즈만큼의 scale zeros를 만들고, 이 크기 만큼 rotation, tx, ty 모두 생성
        scale     = torch.zeros(self.batch, 1, device = self.device) # [B, 1]
        rotation  = torch.zeros_like(scale) # [B, 1]
        tx, ty    = torch.zeros_like(scale), torch.zeros_like(scale) # [B, 1]
        # scale, rot, tx, ty를 모두 dim = 1로 concat해서 [B, 1] * 4 concat -> [B, 4]로 만듬
        params    = self.compose_params(scale, rotation, tx, ty) # [B, 4]

        invalid   = torch.ones_like(scale) # 값이 1인 [B, 1]
        max_trans = torch.ones_like(scale) * self.max_trans # 값이 1인 [B, 1]에 max_trans를 곱함

        while invalid.sum() > 0: # [B, 4].sum() -> B
            # [0.93 ~ 1.0] 범위의 uniform 난수를 각 스케일에 할당
            scale.uniform_(self.min_scale, self.max_scale) # [B, 1]

            # ?????????????????????????????????????
            # 0.5를 곱하는 이유는 translation의 범위는 -0.5에서 0.5이기 때문에 다음의 식을 적용
            # 1인 [12, 1]에서 random [12, 1]를 빼면 범위가 0.0에서 0.07 사이임 0.07보다 크면 0.07을 선택
            # 여기에 0.5를 곱해서 max_t를 계산
            max_t = torch.min(torch.ones_like(scale) - scale, max_trans) * 0.5
            tx    = tx.uniform_(-1.0, 1.0) * max_t * img_size[1]
            ty    = ty.uniform_(-1.0, 1.0) * max_t * img_size[0]

            # scale, rotation, tx, ty를 다시 concat 함 -> rotation은 0인 값
            params_new = self.compose_params(scale, rotation, tx, ty) # [B, 4]
            params     = invalid * params_new + (1 - invalid) * params
            invalid    = self.find_invalid(img_size, params)
        return params


    def calculate_tform_and_grids(self, img_size, resize, params): # 이 코드가 무엇을 하는 코드인지 잘 모르겠음
        intm_scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image
        intm_size_h = torch.floor(img_size[0] * intm_scale)
        intm_size_w = torch.floor(img_size[1] * intm_scale)
        scale_x     = intm_size_w / resize[1]
        scale_y     = intm_size_h / resize[0]

        ## Coord of the resized image
        grid_ww, grid_hh = self.meshgrid(resize[1], resize[0])
        grid_ww          = (grid_ww - (resize[1] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_hh          = (grid_hh - (resize[0] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_pts         = torch.cat([grid_ww, grid_hh, torch.ones_like(grid_hh)], dim=0).unsqueeze(0).expand(self.batch, -1, -1, -1)

        ## 1st - scale_tform -> to intermediate image
        scale_tform          = self.identity(self.batch, self.device)
        scale_tform[:, 0, 0] = scale_x[:, 0]
        scale_tform[:, 1, 1] = scale_y[:, 0]
        pts_tform            = torch.matmul(scale_tform.to(self.device), grid_pts.view(self.batch, 3, -1).to(self.device))

        ## 2st - trans and rotate -> to original image (each pixel contains the coordinates in the original images)
        tr_tform          = self.identity(self.batch, self.device)
        tr_tform[:, 0, 2] = tx[:, 0]
        tr_tform[:, 1, 2] = ty[:, 0]
        pts_tform         = torch.matmul(tr_tform, pts_tform).view(self.batch, 3, resize[0], resize[1])

        grid_img_ww = pts_tform[:, 0, :, :] / float(img_size[1]) * 2    # x2 is for scaling [-1. 1]
        grid_img_hh = pts_tform[:, 1, :, :] / float(img_size[0]) * 2
        grid_img    = torch.cat([grid_img_ww.unsqueeze(3), grid_img_hh.unsqueeze(3)], dim=3)
        return grid_img


    # 이 코드가 무엇을 하는 코드인지 잘 모르겠음
    def augment_intrinsic_matrices(self, intrinsics, num_splits, img_size, resize, params):
        ### Finding the starting pt in the Original Image
        intm_scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image: finding scale from "Resize" to "Intermediate Image"
        intm_size_h = torch.floor(img_size[0] * intm_scale)
        intm_size_w = torch.floor(img_size[1] * intm_scale)
        scale_x     = intm_size_w / resize[1]
        scale_y     = intm_size_h / resize[0]

        ## Coord of the resized image
        pt_o        = torch.zeros([1, 1]).float()
        grid_ww     = (pt_o - (resize[1] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_hh     = (pt_o - (resize[0] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_pts    = torch.cat([grid_ww, grid_hh, torch.ones_like(grid_hh)], dim=0).unsqueeze(0).expand(self.batch, -1, -1, -1)

        ## 1st - scale_tform -> to intermediate image
        scale_tform          = self.identity(self.batch, self.device)
        scale_tform[:, 0, 0] = scale_x[:, 0]
        scale_tform[:, 1, 1] = scale_y[:, 0]
        pts_tform            = torch.matmul(scale_tform.to(self.device), grid_pts.view(self.batch, 3, -1).to(self.device))

        ## 2st - trans and rotate -> to original image (each pixel contains the coordinates in the original images)
        tr_tform          = self.identity(self.batch, self.device)
        tr_tform[:, 0, 2] = tx[:, 0]
        tr_tform[:, 1, 2] = ty[:, 0]
        pts_tform         = torch.matmul(tr_tform, pts_tform)
        str_p_ww          = pts_tform[:, 0, :] + torch.ones_like(pts_tform[:, 0, :]) * float(img_size[1]) * 0.5 
        str_p_hh          = pts_tform[:, 1, :] + torch.ones_like(pts_tform[:, 1, :]) * float(img_size[0]) * 0.5

        ## Cropping
        intrinsics[:, :, 0, 2] -= str_p_ww[:, 0:1].expand(-1, num_splits)
        intrinsics[:, :, 1, 2] -= str_p_hh[:, 0:1].expand(-1, num_splits)

        ## Scaling        
        intrinsics[:, :, 0, 0] = intrinsics[:, :, 0, 0] / scale_x
        intrinsics[:, :, 1, 1] = intrinsics[:, :, 1, 1] / scale_y
        intrinsics[:, :, 0, 2] = intrinsics[:, :, 0, 2] / scale_x
        intrinsics[:, :, 1, 2] = intrinsics[:, :, 1, 2] / scale_y
        return intrinsics




## training에서 사용하는 Augmentation
class AugmentSceneFlow(AugmenScaleCrop):
    def __init__(self, 
                batch, 
                device,
                photometric = True,
                trans = 0.07,
                scale = [0.93, 1.0],
                resize = [256, 832]):
        super(AugmentSceneFlow, self).__init__(batch, device, photometric, trans, scale, resize)
        """
        상속을 받으면 부모 클래스의 arg 인자를 받아서 사용이 가능하고,
        동시에 부모 클래스에서 선언된 메서드들들 사용 가능
        """


    def forward(self, example_dict):
        """
        example_dict
        input_l0, input_l1, input_l2 : [B, 3, 375, 1242]
        input_r0, input_r1, input_r2 : [B, 3, 375, 1242]
        K_l, K_r : [B, C, 3, 3]

        return
        입력 이미지들을 모두 resize [B, 3, H, W] -> [B, 3, 256, 832]
        K_l, K_r를 resize에 맞게 리스케일링하고, 리스케일링 K를 flip한 행렬까지 리턴
        """
        if "input_l0" in example_dict:
           im_l0 = example_dict["input_l0"]
        im_l1 = example_dict["input_l1"]
        im_l2 = example_dict["input_l2"]
        if "input_r0" in example_dict:
            im_r0 = example_dict["input_r0"]
        im_r1 = example_dict["input_r1"]
        im_r2 = example_dict["input_r2"]
        k_l   = example_dict["kl"].clone()
        k_r   = example_dict["kr"].clone()
        B, C, H, W  = im_l1.size() # 이미지의 사이즈 추출
        self.device = im_l1.device # 이미지의 장치 추출

        ## Finding out augmentation parameters
        ## [3, 375, 1242] -> [3, 256, 832] 이미지로 리사이징된 좌표 계산
        params = self.find_aug_params([H, W], self.resize)
        coords = self.calculate_tform_and_grids([H, W], self.resize, params)
        params_scale, _, _, _ = self.decompose_params(params)

        ## Augment images, 이미지를 coords에 맞게 와핑, (375, 1242) -> (256, 832)
        if "input_l0" in example_dict:
            im_l0 = F.grid_sample(im_l0, coords, align_corners = True)
        im_l1 = F.grid_sample(im_l1, coords, align_corners = True)
        im_l2 = F.grid_sample(im_l2, coords, align_corners = True)
        if "input_r0" in example_dict:
            im_r0 = F.grid_sample(im_r0, coords, align_corners = True)
        im_r1 = F.grid_sample(im_r1, coords, align_corners = True)
        im_r2 = F.grid_sample(im_r2, coords, align_corners = True)

        ## Augment intrinsic matrix         
        k_list     = [k_l.unsqueeze(1), k_r.unsqueeze(1)]
        num_splits = len(k_list)
        intrinsics = torch.cat(k_list, dim = 1)
        intrinsics = self.augment_intrinsic_matrices(intrinsics, num_splits, [H, W], self.resize, params)
        k_l, k_r   = torch.chunk(intrinsics, num_splits, dim = 1)
        k_l, k_r   = k_l.squeeze(1), k_r.squeeze(1)

        ## Augmnet images via photometric
        if self.photometric and torch.rand(1) > 0.5:
            if "input_l0" in example_dict:
                im_l0, im_l1, im_l2, im_r0, im_r1, im_r2 = self.photo_aug(
                    im_l0, im_l1, im_l2, im_r0, im_r1, im_r2)
            else: 
                im_l1, im_l2, im_r1, im_r2 = self.photo_aug(im_l1, im_l2, im_r1, im_r2)

        ## [256, 832] 사이즈로 리사이징한 augment 이미지와 augment 카메라 행렬을
        ## 딕셔너리에 업데이트
        example_dict["input_coords"]    = coords
        example_dict["input_aug_scale"] = params_scale
        if "input_l0" in example_dict:
            example_dict["input_l0_aug"] = im_l0
        example_dict["input_l1_aug"] = im_l1
        example_dict["input_l2_aug"] = im_l2
        if "input_r0" in example_dict:
            example_dict["input_r0_aug"] = im_r0
        example_dict["input_r1_aug"] = im_r1
        example_dict["input_r2_aug"] = im_r2
        example_dict["kl_aug"]       = k_l
        example_dict["kr_aug"]       = k_r

        ## resize한 이미지 width에서 - k_l_flip을 뺌 -> flip한 이미지의 intrinsic 계산
        ## 이미지를 좌우 플립하면, intrinsic의 cx = W - cx로 바꾸어서 맞추어 주어야함
        k_l_flip = k_l.clone()
        k_r_flip = k_r.clone()
        k_l_flip[:, 0, 2] = im_l1.size(3) - k_l_flip[:, 0, 2]
        k_r_flip[:, 0, 2] = im_r1.size(3) - k_r_flip[:, 0, 2]
        example_dict["kl_flip_aug"] = k_l_flip
        example_dict["kr_flip_aug"] = k_r_flip

        aug_size       = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self.resize[0]
        aug_size[:, 1] = self.resize[1]
        example_dict["aug_size"] = aug_size
        return example_dict




#3 Validation에서 사용하는 Augmentation (Validation, Test 용)
class AugmentResize(nn.Module):
    def __init__(self, imgsize = [256, 832]):
        super(AugmentResize, self).__init__() # init
        self.imgsize     = imgsize
        self.isRight     = False
        self.photometric = False
        self.photo_augmentation = PhotometricAugmentation()


    def forward(self, example_dict):
        if ('input_r1' in example_dict) and ('input_r2' in example_dict):
            self.isRight = True

        # Focal length rescaling
        _, _, H, W = example_dict["input_l1"].size() # [B, C, H, W]
        # 리사이즈 이미지 / 오리지널 이미지 크기
        sy = self.imgsize[0] / H
        sx = self.imgsize[1] / W

        # Image resizing
        if "input_l0" in example_dict:
            im_l0 = interpolate(example_dict["input_l0"], self.imgsize)
        im_l1 = interpolate(example_dict["input_l1"], self.imgsize)
        im_l2 = interpolate(example_dict["input_l2"], self.imgsize)
        k_l   = intrinsic_scale(example_dict["kl"], sx, sy)

        if self.isRight:
            if "input_r0" in example_dict:
                im_r0 = interpolate(example_dict["input_r0"], self.imgsize)
            im_r1 = interpolate(example_dict["input_r1"], self.imgsize)
            im_r2 = interpolate(example_dict["input_r2"], self.imgsize)
            k_r   = intrinsic_scale(example_dict["kr"], sx, sy)

        # photometric = False라서, augmentation을 하지 않음
        if self.photometric and torch.rand(1) > 0.5:
            if "input_l0" in example_dict:
                if self.isRight:
                    im_l0, im_l1, im_l2, im_r0, im_r1, im_r2 = self.photo_augmentation(
                        im_l0, im_l1, im_l2, im_r0, im_r1, im_r2)
                else:
                    im_l0, im_l1, im_l2 = self.photo_augmentation(im_l0, im_l1, im_l2)
            else:
                if self.isRight:
                    im_l1, im_l2, im_r1, im_r2 = self.photo_augmentation(im_l1, im_l2, im_r1, im_r2)
                else:
                    im_l1, im_l2 = self.photo_augmentation(im_l1, im_l2)


        # 좌측 이미지 어규멘테이션 저장
        if "input_l0" in example_dict:
            example_dict["input_l0_aug"] = im_l0
        example_dict["input_l1_aug"] = im_l1
        example_dict["input_l2_aug"] = im_l2
        example_dict["kl_aug"]      = k_l

        # 우측 이미지 어규멘테이션 저장
        if self.isRight:
            if "input_r0" in example_dict:
                example_dict["input_r0_aug"] = im_r0
            example_dict["input_r1_aug"] = im_r1
            example_dict["input_r2_aug"] = im_r2
            example_dict["kr_aug"]      = k_r

        # 좌측 intrinsic을 flip한 것을 저장
        k_l_flip          = k_l.clone()
        k_l_flip[:, 0, 2] = im_l1.size(3) - k_l_flip[:, 0, 2]
        example_dict["kl_flip_aug"] = k_l_flip

        # 우측 intrinsic을 flip한 것을 저장
        if self.isRight:
            k_r_flip          = k_r.clone()
            k_r_flip[:, 0, 2] = im_r1.size(3) - k_r_flip[:, 0, 2]
            example_dict["kr_flip_aug"] = k_r_flip

        aug_size       = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self.imgsize[0]
        aug_size[:, 1] = self.imgsize[1]
        example_dict["aug_size"] = aug_size
        return example_dict