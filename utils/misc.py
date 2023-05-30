import torch
import numpy as np

def nms(dets, scores, nms_thresh=0.4):
    x1 = dets[:, 0]  #xmin
    y1 = dets[:, 1]  #ymin
    x2 = dets[:, 2]  #xmax
    y2 = dets[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
    order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

    keep = []                                             # store the final bounding boxes
    while order.size > 0:
        i = order[0]                                      #the index of the bbox with highest confidence
        keep.append(i)                                    #save it to keep
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

def dvs_detection_collate(batch):
    targets = []
    ori_targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        ori_targets.append(sample[2])
    return torch.from_numpy(np.stack(imgs, 0)), targets, ori_targets
