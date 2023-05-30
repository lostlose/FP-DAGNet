import argparse
import numpy as np
import tqdm
import torch

from config.anchor_config import gen1_cfg
from data.gen1 import Resize_frame, Gen1_sbn, Gen1_sbt

from utils import create_labels, tools
from utils.misc import dvs_detection_collate
from evaluator.gen1_evaluate import coco_eval


parser = argparse.ArgumentParser(description='DVS Detection')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='img_size')
parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
parser.add_argument('--show', action='store_true', default=False,
                    help='show the visulization results.')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
parser.add_argument('--experiment_description', type=str, default='no description',
                    help='describ the experiment')
# model
parser.add_argument('--weight', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.5, type=float,
                    help='NMS threshold')
parser.add_argument('--nms_thresh', default=0.5, type=float,
                    help='NMS threshold')
parser.add_argument('--center_sample', action='store_true', default=False,
                    help='center sample trick.')
parser.add_argument('-t', '--time_steps', default=5, type=int,
                    help='snn time steps')
parser.add_argument('-ef', '--events_per_frame', default=25000, type=int,
                    help='sbn events num per frame')
parser.add_argument('-tf', '--time_per_frame', default=10, type=int,
                    help='sbt ms per frame')
parser.add_argument('-fs', '--frame_per_stack', default=5, type=int,
                    help='frames per stack')
parser.add_argument('-b', '--spike_b', default=3, type=int,
                    help='spike b')
parser.add_argument('--bn', action='store_false', default=True, 
                    help='use bn layer')
parser.add_argument('--frame_method', default='sbn', type=str,
                        help='sbt or sbn')

# dataset
parser.add_argument('--root', default='./gen1/',
                    help='data root')
parser.add_argument('-d', '--dataset', default='gen1',
                    help='gen1.')
parser.add_argument('--multi_anchor', action='store_true', default=False,
                        help='use multiple anchor boxes as the positive samples')

# architecture params
parser.add_argument('--fea_num_layers', type=int, default=4)
parser.add_argument('--fea_filter_multiplier', type=int, default=8)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)

args = parser.parse_args()

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:3]
    else:
        this_str = this_str.split('.')
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append('[%s]'%value)
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append('.'+value)
    return ''.join(new_index)

if __name__ == '__main__':
    args = parser.parse_args()
    # device
    if args.device != 'cpu':
        print('use cuda:{}'.format(args.device))
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")

    print('Model:FP-DAGNet')

    # dataset and evaluator
    if args.frame_method == 'sbt':
        test_dataset = Gen1_sbt(args.root, object_classes='all', height=240, width=304,
                                mode='test',
                                ms_per_frame=args.time_per_frame, frame_per_sequence=args.frame_per_stack,
                                T=args.time_steps, transform=Resize_frame(args.img_size), sbt_method='before')
    else:
        test_dataset = Gen1_sbn(args.root, object_classes='all', height=240, width=304,
                                mode='test',
                                events_per_frame=args.events_per_frame, frame_per_sequence=args.frame_per_stack,
                                T=args.time_steps, transform=Resize_frame(args.img_size), sbn_method='before')
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dvs_detection_collate,
        num_workers=args.num_workers,
        pin_memory=True
    )
    classes_name = test_dataset.object_classes
    num_classes = len(classes_name)
    # np.random.seed(0)

    # Gen1 Config
    cfg = gen1_cfg
    # build model
    from models.network import FP_DAGNet as net

    model = net(device=device,
                     input_size=args.img_size,
                     num_classes=num_classes,
                     trainable=False,
                     cfg=cfg,
                     center_sample=args.center_sample,
                     time_steps=args.time_steps,
                     spike_b=args.spike_b,
                     bn=args.bn,
                     init_channels=args.frame_per_stack,
                     args=args)
    anchor_size = model.anchor_list
    all_keys = [convert_str2index(name, is_cell=True) for name, value in model.named_parameters() if '_ops' in name]
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval('model.%s.mem' % key)
            mem_keys.append(key)
        except:
            print(key)
            pass
    print('mem', mem_keys)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    model.set_mem_keys(mem_keys)
    print('Finished loading model!')

    gt_label_list = []
    pred_label_list = []
    idx2class = ['Car', "Pedestrian"]
    idx2color = ['red', 'green']
    with torch.no_grad():
        for id_, data in enumerate(tqdm.tqdm(test_dataloader)):
            # print('reset all mem')
            # for key in mem_keys:
            #     exec('model.%s.mem=None' % key)
            image, targets, original_label = data
            for label in original_label:
                gt_label_list.append(label)
            targets = [label.tolist() for label in targets]
            size = np.array([[image.shape[-1], image.shape[-2],
                              image.shape[-1], image.shape[-2]]])
            targets = create_labels.gt_creator(
                img_size=args.img_size,
                strides=model.stride,
                label_lists=targets,
                anchor_size=anchor_size,
                multi_anchor=args.multi_anchor,
                center_sample=args.center_sample)
            # to device
            image = image.float().to(device)

            # forward
            conf_pred, cls_pred, reg_pred, box_pred = model(image)

            bboxes, scores, cls_inds = tools.get_box_score(conf_pred, cls_pred, box_pred,
                                                           num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
            bboxes = [box * size for box in bboxes]
            bboxes = [tools.resized_box_to_original(box, args.img_size, test_dataset.height, test_dataset.width) for box in bboxes]
            for i in range(len(bboxes)):
                pred_label = []
                for j, (box, score, cls_ind) in enumerate(zip(bboxes[i], scores[i], cls_inds[i])):
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])

                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(score) # object score * class score
                    A = {"image_id": id_ * args.batch_size + i, "category_id": cls_ind, "bbox": bbox,
                        "score": score} # COCO json format
                    pred_label.append(A)
                pred_label_list.append(pred_label)
    map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=240, width=304, labelmap=classes_name)
    print('test mAP(0.5:0.95):{}, mAP(0.5):{}'.format(map50_95, map50))