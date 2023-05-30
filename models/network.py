import torch
import torch.nn as nn
import numpy as np
from models.new_model_2d import newFeature
from models.SNN import SNN_2d, SNN_2d_lsnn

class FP_DAGNet(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 cfg=None, center_sample=False, bn=True, init_channels=5, time_steps=5, spike_b=3, args=None):
        super(FP_DAGNet, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.time_steps = time_steps

        network_path_fea = np.array(eval(args.net_arch_fea))
        cell_arch_fea = np.array(eval(args.cell_arch_fea))

        # backbone
        self.feature = newFeature(init_channels, network_path_fea, cell_arch_fea, args=args)
        self.stride = self.feature.stride
        num_out = len(self.stride)
        anchor_size = cfg['anchor_size_gen1_{}'.format(num_out * 3)]
        self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)

        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        out_channel = 2 * args.fea_block_multiplier * args.fea_filter_multiplier

        # pred
        num_out = len(self.stride)
        if num_out == 1:
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        elif num_out == 2:
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        else:
            self.head_det_1 = nn.Conv2d(out_channel * 4, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

    def create_grid(self, input_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred, index):
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy[index]) * self.stride[index]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy[index]) * self.stride[index]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh[index]
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def set_mem_keys(self, mem_keys):
        self.mem_keys = mem_keys

    def clear_mem(self):
        for key in self.mem_keys:
            exec('self.%s.mem=None' % key)
        for m in self.modules():
            if isinstance(m, SNN_2d) or isinstance(m, SNN_2d_lsnn):
                m.mem = None

    def forward(self, x):
        self.clear_mem()
        C = self.num_classes
        param = {'mixed_at_mem':True, 'is_first':False}
        B, T, c, H, W = x.shape
        # x = x[:,1:]
        # x = x.reshape(B, 3, 3, H, W)

        for t in range(self.time_steps):
            if t == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            y = self.feature(x[:, t, ...],param)
        num_out = len(self.stride)
        if num_out == 1:
            y3 = y
            pred_s = self.head_det_3(y3)
            preds = [pred_s]
        elif num_out == 2:
            y2, y3 = y
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m]
        else:
            y1, y2, y3 = y
            pred_l = self.head_det_1(y1)
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m, pred_l]

        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred





