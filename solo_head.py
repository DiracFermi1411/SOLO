from backbone import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
import skimage.transform
import os.path
import shutil

import gc
gc.enable()


class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384),
                               (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                    alpha=0.25,
                                    weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        # TODO initialize layers: stack intermediate layer and output layer
        num_groups = 32

        self.cate_head = nn.ModuleList([])
        for i in range(self.stacked_convs):
            layer = nn.ModuleList([
                nn.Conv2d(self.in_channels, self.seg_feat_channels,
                          kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, self.seg_feat_channels),
                nn.ReLU(True)
            ])
            self.cate_head.append(layer)

        self.cate_out = nn.ModuleList([
            nn.Conv2d(self.seg_feat_channels, self.cate_out_channels,
                      kernel_size=(3, 3), padding=1, bias=True),
            nn.Sigmoid()
        ])

        self.ins_head = nn.ModuleList([])
        first_layer = nn.ModuleList([
            nn.Conv2d(self.in_channels+2, self.seg_feat_channels,
                      kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, self.seg_feat_channels),
            nn.ReLU(True)
        ])
        self.ins_head.append(first_layer)
        for i in range(self.stacked_convs-1):
            layer = nn.ModuleList([
                nn.Conv2d(self.in_channels, self.seg_feat_channels,
                          kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, self.seg_feat_channels),
                nn.ReLU(True)
            ])
            self.ins_head.append(layer)

        num_grids_2 = list(map(lambda x: pow(x, 2), self.seg_num_grids))

        ins_out_list_tmp = []           # python list
        last_layer = nn.ModuleList([])
        for i in range(len(num_grids_2)):
            last_layer = nn.ModuleList([
                nn.Conv2d(self.seg_feat_channels, num_grids_2[i], kernel_size=(
                    1, 1), padding=0, bias=True),
                nn.Sigmoid()
            ])
            ins_out_list_tmp.append(last_layer)
        # convert to nn.ModuleList for GPU capability
        self.ins_out_list = nn.ModuleList(ins_out_list_tmp)

    # This function initialize weights for head network

    def _init_weights(self):
        # TODO: initialize the weights
        for m in self.cate_head.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data)
                assert m.bias is None
        for m in self.ins_head.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data)
                assert m.bias is None
        for m in self.cate_out.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)
        for layer in self.ins_out_list:
            for m in layer:
                if isinstance(m, nn.Conv2d):
                    m.weight.data = torch.nn.init.xavier_uniform_(
                        m.weight.data)
                    nn.init.constant_(m.bias, 0)

    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling

    def forward(self,
                fpn_feat_list,
                eval):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256, 100, 136)
        quart_shape = [new_fpn_list[0].shape[-2]*2,
                       new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level,
                                                        new_fpn_list, list(
                                                            range(len(new_fpn_list))),
                                                        eval=eval, upsample_shape=quart_shape)
        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        new_fpn_list = [
            torch.nn.functional.interpolate(
                fpn_feat_list[0], scale_factor=1/2),
            fpn_feat_list[1],
            fpn_feat_list[2],
            fpn_feat_list[3],
            F.interpolate(fpn_feat_list[4], size=(25, 34))
        ]
        return new_fpn_list

    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==True
        # cate_pred: (bz,S,S,C-1) / after point_NMS
        # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
        # if eval==False
        # cate_pred: (bz,C-1,S,S)
        # ins_pred: (bz, S^2, 2H_feat, 2W_feat)

    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        # TODO: finish forward function for single level in FPN.
        # Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid

        def gridmap(H_feat, W_feat, device):
            x = torch.linspace(-1, 1, H_feat)
            y = torch.linspace(-1, 1, W_feat)
            xm, ym = torch.meshgrid([x, y])
            xm = torch.unsqueeze(xm, 0)
            ym = torch.unsqueeze(ym, 0)
            xm = torch.unsqueeze(xm, 0)
            ym = torch.unsqueeze(ym, 0)
            xm = xm.repeat(bz, 1, 1, 1)
            ym = ym.repeat(bz, 1, 1, 1)
            xm, ym = xm.to(device), ym.to(device)
            return xm, ym
        # upsample_shape=200，272

        # in inference time, upsample the pred to (ori image size/4)
        H_feat = fpn_feat.shape[2]
        W_feat = fpn_feat.shape[3]
        bz = fpn_feat.shape[0]
        fnp_idx = self.ins_out_list[idx]
        H = int(upsample_shape[0]/2)  # 100
        W = int(upsample_shape[1]/2)  # 136

        if eval == True:
            # TODO resize ins_pred
            cate_pred = F.interpolate(fpn_feat, size=(
                num_grid, num_grid), mode='bilinear')
            for y in self.cate_head:
                for f in y:
                    cate_pred = f(cate_pred)
            for f in self.cate_out:
                cate_pred = f(cate_pred)

            cate_pred = self.points_nms(cate_pred)

            xm, ym = gridmap(H_feat, W_feat, ins_pred.device)
            ins_pred = torch.cat((ins_pred, xm), dim=1)
            ins_pred = torch.cat((ins_pred, ym), dim=1)
            for y in self.ins_head:
                for f in y:
                    ins_pred = f(ins_pred)
            ins_pred = F.interpolate(
                ins_pred, size=(2*H, 2*W))
            for f in fnp_idx:
                ins_pred = f(ins_pred)
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, 200, 272)

        if eval == False:
            cate_pred = F.interpolate(fpn_feat, size=(
                num_grid, num_grid), mode='bilinear')
            for f in self.cate_head:
                for y in f:
                    cate_pred = y(cate_pred)
            for f in self.cate_out:
                cate_pred = f(cate_pred)
            xm, ym = gridmap(H_feat, W_feat, ins_pred.device)
            ins_pred = torch.cat((ins_pred, xm), dim=1)
            ins_pred = torch.cat((ins_pred, ym), dim=1)
            for y in self.ins_head:
                for f in y:
                    ins_pred = f(ins_pred)

            ins_pred = F.interpolate(ins_pred, size=(
                2*H_feat, 2*W_feat))
            for f in fnp_idx:
                ins_pred = f(ins_pred)

            # check flag
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (
                num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        # TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        fpn = len(ins_gts_list[0])
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                              for ins_labels_level_img, ins_ind_labels_level_img in
                              zip(ins_labels_level, ins_ind_labels_level)], 0)
                   for ins_labels_level, ins_ind_labels_level in
                   zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]
        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in
                     zip(ins_pred_list, zip(*ins_ind_gts_list))]

        cate_gts = [torch.cat([cate_gts_level_img.flatten() for cate_gts_level_img in cate_gts_level])
                    for cate_gts_level in zip(*cate_gts_list)]

        cate_gts = torch.cat(cate_gts)

        cate_preds = [cate_pred_level.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
                      for cate_pred_level in cate_pred_list]
        cate_preds = torch.cat(cate_preds, 0)

        cate_loss = self.FocalLoss(cate_preds, cate_gts)

        s_total = 0
        n_total = 0
        for layer_idx in range(fpn):
            N_pos = ins_gts[layer_idx].shape[0]
            if N_pos == 0:
                continue
            active_mask_pred = ins_preds[layer_idx]
            active_mask_target = ins_gts[layer_idx]
            d_mask = map(self.DiceLoss, active_mask_pred, active_mask_target)
            d_mask = list(d_mask)
            d_mask_sum = sum(d_mask)
            s_total += d_mask_sum
            n_total += N_pos
        mask_loss = s_total/(n_total+1e-9)

        total_loss = cate_loss+self.mask_loss_cfg["weight"]*mask_loss

        return cate_loss, mask_loss, total_loss

    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar

    def DiceLoss(self, mask_pred, mask_gt):
        # TODO: compute DiceLoss
        numerator = 2*torch.sum(mask_pred*mask_gt)
        denominator = torch.sum(
            torch.pow(mask_pred, 2) + torch.pow(mask_gt, 2)) + 1e-9
        dice = numerator/denominator
        return (1-dice)

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        # TODO: compute focalloss
        alpha = self.cate_loss_cfg['alpha']
        gamma = self.cate_loss_cfg['gamma']
        N = cate_preds.shape[0]
        C = cate_preds.shape[1]+1
        cate_preds = cate_preds.flatten()

        idx_row = list(np.arange(0, N))
        idx_col = list(cate_gts.long().cpu().numpy())
        one_hot_raw = torch.zeros(
            (N, C), device=cate_preds.device, dtype=torch.long)
        one_hot_raw[idx_row, idx_col] = 1
        one_hot = one_hot_raw[:, 1:]
        one_hot = one_hot.flatten()

        m = one_hot.shape[0]
        inv_one_hot = 1-one_hot
        p2 = cate_preds*inv_one_hot
        p1 = 1-p2
        p3 = p1+cate_preds-one_hot
        p4 = 1-p3
        y = -torch.sum((1-alpha) * torch.pow(p2, gamma) * torch.log(p1))
        y_1 = -torch.sum(alpha * torch.pow(p4, gamma) * torch.log(p3))
        focal_loss = (y_1 + y)/(m + 1e-9)

        return focal_loss

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training
        feature_sizes = [(ins_pred.shape[2], ins_pred.shape[3])
                         for ins_pred in ins_pred_list]
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.target_single_img,
                                                                        bbox_list, label_list, mask_list,
                                                                        featmap_sizes=feature_sizes)

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (
            self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    # process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )

    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        # TODO: finish single image target build
        # compute the area of every object in this single image
        device = gt_bboxes_raw.device

        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        N_obj, H_ori, W_ori = gt_masks_raw.shape
        obj_scale_list = []
        obj_center_list = []
        obj_center_regions = []

        obj_indice = [list() for i in range(len(featmap_sizes))]

        for obj_idx in range(N_obj):
            bbox = gt_bboxes_raw[obj_idx]
            obj_w, obj_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            obj_scale = torch.sqrt((obj_w * W_ori) * (obj_h * H_ori))
            obj_scale_list.append(obj_scale)
            assert obj_scale > 1
            for level_idx, level_range in enumerate(self.scale_ranges, 0):
                if (obj_scale >= level_range[0]) and (obj_scale < level_range[1]):
                    obj_indice[level_idx].append(obj_idx)

            obj_c_y, obj_c_x = ndimage.center_of_mass(
                gt_masks_raw[obj_idx].cpu().numpy())
            obj_c_y, obj_c_x = obj_c_y / H_ori, obj_c_x / \
                W_ori
            obj_center_list.append(torch.tensor(
                [obj_c_x, obj_c_y], dtype=torch.float))
            obj_center_regions.append(torch.tensor([
                obj_c_x - self.epsilon * obj_w / 2.0,
                obj_c_y - self.epsilon * obj_h / 2.0,
                obj_c_x + self.epsilon * obj_w / 2.0,
                obj_c_y + self.epsilon * obj_h / 2.0,
            ], dtype=torch.float))

        for level_idx, feat_size in enumerate(featmap_sizes, 0):
            S = self.seg_num_grids[level_idx]
            assert len(feat_size) == 2
            cate_label_map = torch.zeros(
                (S, S), dtype=torch.long, device=device)
            ins_label_map = torch.zeros(
                (S * S, feat_size[0], feat_size[1]), dtype=torch.long, device=device)
            ins_ind_label = torch.zeros(
                (S * S), dtype=torch.bool, device=device)

            for obj_idx in obj_indice[level_idx]:
                obj_region_i = torch.floor(obj_center_regions[obj_idx] * S)
                obj_center_i = torch.floor(obj_center_list[obj_idx] * S)

                x_min = max(obj_center_i[0].item() - 1, obj_region_i[0].item())
                y_min = max(obj_center_i[1].item() - 1, obj_region_i[1].item())
                x_max = min(obj_center_i[0].item() + 1, obj_region_i[2].item())
                y_max = min(obj_center_i[1].item() + 1, obj_region_i[3].item())

                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, S-1), max(y_max, S-1)

                x_min, y_min, x_max, y_max = int(x_min), int(
                    y_min), int(x_max), int(y_max)

                cate_label_map[y_min: (
                    y_max + 1), x_min: (x_max + 1)] = gt_labels_raw[obj_idx]

                mask_raw = gt_masks_raw[obj_idx: (obj_idx+1)]
                mask_resized = BuildDataset.interp2d(
                    mask_raw, feat_size[0], feat_size[1])
                for i in range(y_min, y_max + 1):
                    for j in range(x_min, x_max + 1):
                        ins_label_map[i * S + j] = mask_resized
                        ins_ind_label[i * S + j] = True

            cate_label_list.append(cate_label_map)
            ins_label_list.append(ins_label_map)
            ins_ind_label_list.append(ins_ind_label)

        # check flag
        assert ins_label_list[1].shape == (1296, 200, 272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        # TODO: finish PostProcess
        bz = ins_pred_list[0].shape[0]
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        N_fpn = len(ins_pred_list)
        assert N_fpn == len(cate_pred_list)
        for img_i in range(bz):
            ins_pred_img = torch.cat([ins_pred_list[i][img_i]
                                     for i in range(N_fpn)], dim=0)

            tmp_list = []
            for fpn_i in range(N_fpn):
                cate_pred = cate_pred_list[fpn_i][img_i]
                C, S_1, S_2 = cate_pred.shape
                assert cate_pred.shape[1] == cate_pred.shape[2]
                tmp_x = cate_pred.view(C, S_1 * S_2)
                tmp_list.append(tmp_x.permute(1, 0))
            cate_pred_img = torch.cat(tmp_list, dim=0)
            assert cate_pred_img.shape[1] == 3

            NMS_sorted_scores, NMS_sorted_cate_label, NMS_sorted_ins = self.PostProcessImg(
                ins_pred_img, cate_pred_img, ori_size)
            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_label)
            NMS_sorted_ins_list.append(NMS_sorted_ins)

        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list

    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        # TODO: PostProcess on single image.
        scores_raw, labels_raw = torch.max(cate_pred_img, dim=1)
        cate_indicator = scores_raw > self.postprocess_cfg['cate_thresh']
        scores_raw = scores_raw * cate_indicator

        indicator_map = ins_pred_img > self.postprocess_cfg['ins_thresh']
        sum_1 = torch.sum(ins_pred_img * indicator_map, dim=(1, 2))
        sum_2 = torch.sum(indicator_map, dim=(1, 2)) + 1e-8
        coeff = sum_1 / sum_2
        assert coeff.ndim == 1
        assert coeff.shape[0] == ins_pred_img.shape[0]
        scores = coeff * scores_raw

        _, sorted_indice = torch.sort(scores, descending=True)
        sorted_indice = sorted_indice[0:self.postprocess_cfg['pre_NMS_num']]
        assert len(sorted_indice) == self.postprocess_cfg['pre_NMS_num']

        sorted_score = scores[sorted_indice]
        sorted_label = labels_raw[sorted_indice]
        sorted_ins_bin = indicator_map[sorted_indice]       # hard binary mask
        sorted_ins = ins_pred_img[sorted_indice]

        scores_nms = self.MatrixNMS(sorted_ins_bin, sorted_score)

        _, max_indice = torch.sort(scores_nms, descending=True)
        max_indice = max_indice[0:self.postprocess_cfg['keep_instance']]
        NMS_sorted_scores = scores_nms[max_indice]

        NMS_sorted_cate_label = sorted_label[max_indice] + 1

        resized_mask = torch.nn.functional.interpolate(
            sorted_ins[max_indice].unsqueeze(0), scale_factor=(4, 4))
        resized_mask = resized_mask.squeeze(0)
        NMS_sorted_ins = resized_mask

        high_prob_indice = NMS_sorted_scores > 0.0

        return NMS_sorted_scores[high_prob_indice], NMS_sorted_cate_label[high_prob_indice], NMS_sorted_ins[high_prob_indice]

    def IOU(self, sorted_ins_1, sorted_ins_2):
        assert sorted_ins_1.dtype == torch.bool
        sorted_ins_1 = sorted_ins_1 * 1.0
        sorted_ins_2 = sorted_ins_2 * 1.0
        N, H, W = sorted_ins_1.shape
        M = sorted_ins_2.shape[0]
        assert sorted_ins_2.shape[1] == H
        assert sorted_ins_2.shape[2] == W

        ins_flatten_1 = sorted_ins_1.view(N, H * W)
        ins_flatten_2 = sorted_ins_2.view(M, H * W)

        intersection = torch.matmul(
            ins_flatten_1, ins_flatten_2.transpose(0, 1))
        mask_areas_1 = torch.sum(
            ins_flatten_1, dim=1)
        mask_areas_1 = mask_areas_1.expand(
            M, N).transpose(0, 1)
        mask_areas_2 = torch.sum(
            ins_flatten_2, dim=1)
        mask_areas_2 = mask_areas_2.expand(
            N, M)
        union = mask_areas_1 + mask_areas_2 - intersection + 1e-8
        iou = intersection / union
        return iou

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins: torch.Tensor, sorted_scores, method='gauss', gauss_sigma=0.5):
        # TODO: finish MatrixNMS

        N_act, _, _ = sorted_ins.shape

        iou = self.IOU(sorted_ins, sorted_ins)
        iou = iou.triu(diagonal=1)

        iou_max, _ = torch.max(iou, dim=0)
        iou_max = iou_max.expand(N_act, N_act).transpose(
            0, 1)

        decay, _ = torch.min(
            torch.where(method == 'gauss', torch.exp(-1 * torch.pow(iou, 2) / gauss_sigma), 1 - iou) /
            torch.where(method == 'gauss', torch.exp(-1 *
                        torch.pow(iou_max, 2) / gauss_sigma), 1 - iou_max),
            dim=0
        )

        decay_scores = sorted_scores * decay
        return decay_scores

    # -----------------------------------
    # The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        # self.strides: [8,8,16,32,32]

    def PlotGT(self,
               ins_gts_list,  # mask_label_list
               ins_ind_gts_list,  # mask_index_label_list
               cate_gts_list,  # category_label_list
               color_list,
               img):
        # TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        # This is an important visual check flag.

        rgb_color_list = []
        for color_str in color_list:
            color_map = cm.ScalarMappable(cmap=color_str)
            rgb_value = np.array(color_map.to_rgba(0))[:3]
            rgb_color_list.append(rgb_value)

        for img_i in range(len(ins_gts_list)):
            img_single = img[img_i]
            for level_i in range(len(ins_gts_list[img_i])):
                ins_gts = ins_gts_list[img_i][level_i]
                cate_gts = cate_gts_list[img_i][level_i]
                ins_ind_gts = ins_ind_gts_list[img_i][level_i]

                assert ins_gts.shape[1] % 2 == 0
                assert ins_gts.shape[2] % 2 == 0
                H_feat = int(ins_gts.shape[1] / 2)
                W_feat = int(ins_gts.shape[2] / 2)
                S = cate_gts.shape[0]

                mask_vis = np.zeros((2 * H_feat, 2 * W_feat, 3))
                for flatten_tensor in torch.nonzero(ins_ind_gts, as_tuple=False):
                    flatten_idx = flatten_tensor.item()
                    grid_i = int(flatten_idx / S)
                    grid_j = flatten_idx % S
                    obj_label = cate_gts[grid_i, grid_j]
                    assert obj_label != 0.0

                    rgb_color = rgb_color_list[obj_label - 1]
                    obj_mask = ins_gts[flatten_idx].cpu().numpy()
                    obj_mask_3 = np.stack(
                        [obj_mask, obj_mask, obj_mask], axis=2)
                    mask_vis = mask_vis + obj_mask_3 * rgb_color

                mask_vis_resized = skimage.transform.resize(
                    mask_vis, (img_single.shape[1], img_single.shape[2], 3))

                img_vis = img_single.cpu().numpy().transpose((1, 2, 0))
                img_vis = mask_vis_resized + img_vis * (mask_vis_resized == 0)

                fig, ax = plt.subplots(1)
                ax.imshow(img_vis)
                plt.show()

                img_id = 1
                file_name = f"plotgt_result/img_{img_id}_fpn_{level_i}.png"
                while os.path.isfile(file_name):
                    img_id = img_id + 1
                    file_name = f"plotgt_result/img_{img_id}_fpn_{level_i}.png"
                plt.savefig(file_name)

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        # TODO: Plot predictions
        rgb_color_list = []
        for color_str in color_list:
            color_map = cm.ScalarMappable(cmap=color_str)
            rgb_value = np.array(color_map.to_rgba(0))[:3]
            rgb_color_list.append(rgb_value)

        for img_i, data in enumerate(zip(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list, img), 0):
            score, cate_label, ins, img_single = data
            img_vis = img_single.cpu().numpy().transpose((1, 2, 0))

            fig, ax = plt.subplots(1)
            ax.imshow(img_vis)
            plt.show()
            os.makedirs("infer_result", exist_ok=True)
            file_name = f"infer_result/batch_{iter_ind}_img_{img_i}_ori.png"
            plt.savefig(file_name)

            mask_vis = np.zeros_like(img_vis)
            for ins_id in range(len(score)):
                obj_label = cate_label[ins_id]
                ins_bin = (ins[ins_id] >=
                           self.postprocess_cfg['ins_thresh']) * 1.0
                obj_mask = ins_bin.cpu().numpy()

                rgb_color = rgb_color_list[obj_label - 1]
                obj_mask_3 = np.stack(
                    [obj_mask, obj_mask, obj_mask], axis=2)
                mask_vis = mask_vis + obj_mask_3 * rgb_color

            img_vis = mask_vis + img_vis * (mask_vis == 0)

            fig, ax = plt.subplots(1)
            ax.imshow(img_vis)
            plt.show()

            os.makedirs("infer_result", exist_ok=True)
            file_name = f"infer_result/batch_{iter_ind}_img_{img_i}.png"
            plt.savefig(file_name)


if __name__ == '__main__':
    # file path and make a list
    imgs_path = '../../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../../data/hw3_mycocodata_bboxes_comp_zlib.npy'

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    resnet50_fpn = Resnet50Backbone()
    # class number is 4, because consider the background as one category.
    solo_head = SOLOHead(num_classes=4)
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50_fpn = resnet50_fpn.to(device)
    solo_head = solo_head.to(device)

    try:
        shutil.rmtree("plotgt_result")
    except FileNotFoundError:
        pass
    os.makedirs("plotgt_result", exist_ok=True)

    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [
            data[i] for i in range(len(data))]
        # fpn is a dict
        img = img.to(device)
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        # demo
        cate_pred_list, ins_pred_list = solo_head.forward(
            fpn_feat_list, eval=False)

        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)

        cate_loss, mask_loss, total_loss = solo_head.loss(
            cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
