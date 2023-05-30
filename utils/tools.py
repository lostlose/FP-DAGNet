import numpy as np
import torch

def postprocess(bboxes, scores, num_classes, conf_thresh, nms_thresh):
    B = len(bboxes)
    total_bboxes = []
    total_scores = []
    total_cls_inds = []
    for i in range(B):
        score = scores[i]
        bbox = bboxes[i]
        cls_ind = np.argmax(score, axis=1)
        score = score[(np.arange(score.shape[0]), cls_ind)]

        # threshold
        # print(score.shape)
        keep = np.where(score >= conf_thresh)
        bbox = bbox[keep]
        score = score[keep]
        # print(score.shape)
        cls_ind = cls_ind[keep]

        # NMS
        keep = np.zeros(len(bbox), dtype=np.int)
        for j in range(num_classes):
            inds = np.where(cls_ind == j)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox[inds]
            c_scores = score[inds]
            c_keep = nms(c_bboxes, c_scores, nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox = bbox[keep]
        score = score[keep]
        # print(score.shape)
        cls_ind = cls_ind[keep]

        total_bboxes.append(bbox)
        total_scores.append(score)
        total_cls_inds.append(cls_ind)

    return total_bboxes, total_scores, total_cls_inds

def nms(dets, scores, nms_thresh):
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)  # the size of bbox
    order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

    keep = []  # store the final bounding boxes
    while order.size > 0:
        i = order[0]  # the index of the bbox with highest confidence
        keep.append(i)  # save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def get_box_score(conf_pred, cls_pred, box_pred, num_classes, conf_thresh=0.01, nms_thresh=0.5):
    # conf_pred = torch.sigmoid(conf_pred)
    # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
    bboxes = torch.clamp(box_pred, 0., 1.)
    # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C],
    scores = torch.softmax(cls_pred, dim=-1) * torch.sigmoid(conf_pred)
    # scores = torch.softmax(cls_pred, dim=-1)

    # 将预测放在cpu处理上，以便进行后处理
    scores = scores.to('cpu').numpy()
    bboxes = bboxes.to('cpu').numpy()

    # 后处理
    bboxes, scores, cls_inds = postprocess(bboxes, scores, num_classes=num_classes, conf_thresh=conf_thresh, nms_thresh=nms_thresh)

    return bboxes, scores, cls_inds

def resized_box_to_original(resized_box, input_size, h_original, w_original):
    r = min(input_size / h_original, input_size / w_original)
    h_resize, w_resize = int(round(r * h_original)), int(round(r * w_original))
    h_pad, w_pad = (input_size - h_resize) / 2, (input_size - w_resize) / 2
    offset = np.array([w_pad, h_pad, w_pad, h_pad])
    original_label = resized_box - offset
    original_label = original_label / r
    return original_label
