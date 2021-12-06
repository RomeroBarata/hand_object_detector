# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import json
import numpy as np
import os
import pdb
import pprint
import sys
import time

import cv2
from PIL import Image
# from scipy.misc import imread
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.matching import filter_object
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL
from model.utils.net_utils import vis_detections_filtered_objects_PIL, vis_detections_filtered_objects
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results',
                        default="images_det")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=89999, type=int, required=True)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)

    parsed_args = parser.parse_args()
    return parsed_args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def get_det_info(dets, det_type='objs'):
    hand_id_to_hand_name = {0: 'Left', 1: 'Right'}
    state_id_to_state_name = {0: 'No Contact', 1: 'Self Contact', 2: 'Another Person',
                              3: 'Portable Object', 4: 'Stationary Object'}
    bboxes, scores = [], []
    for det in dets:
        bbox = list(int(np.round(x)) for x in det[:4])
        bboxes.append(bbox)
        score = det[4].item()
        scores.append(score)
    det_info = {'bboxes': bboxes, 'scores': scores}
    if det_type == 'hands':
        hands, states = [], []
        for det in dets:
            hand = hand_id_to_hand_name[det[-1]]
            hands.append(hand)
            state = state_id_to_state_name[det[5]]
            states.append(state)
        det_info['hands'] = hands
        det_info['states'] = states
    return det_info


if __name__ == '__main__':
    args = parse_args()

    video_id = os.path.basename(args.image_dir)
    save_dir = os.path.join(args.save_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)
    save_json_filepath = os.path.join(save_dir, video_id + '.json')
    if os.path.isfile(save_json_filepath):
        sys.exit(f'Hand-object detections have already been extracted for video {video_id}. Skipping it.')
    save_imgs_dir = os.path.join(save_dir, 'vis')
    os.makedirs(save_imgs_dir, exist_ok=True)
    # print('Called with args:')
    # print(args)
    font_path = os.path.join(os.path.dirname(args.load_dir), 'lib/model/utils/times_b.ttf')

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

    # initialize the network
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % load_name)
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initialize tensor holders
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand
        thresh_obj = args.thresh_obj
        vis = args.vis

        # print(f'thresh_hand = {thresh_hand}')
        # print(f'thnres_obj = {thresh_obj}')

        webcam_num = args.webcam_num
        # Set up webcam or get image directories
        if webcam_num >= 0:
            cap = cv2.VideoCapture(webcam_num)
            num_images = 0
        else:
            print(f'image dir = {args.image_dir}')
            print(f'save dir = {args.save_dir}')
            imglist = sorted(os.listdir(args.image_dir), reverse=True)
            num_images = len(imglist)

        print('Loaded Photo: {} images.'.format(num_images))
        data_to_export = []
        while num_images > 0:
            total_tic = time.time()
            if webcam_num == -1:
                num_images -= 1

            # Get image from the webcam
            if webcam_num >= 0:
                if not cap.isOpened():
                    raise RuntimeError("Webcam could not open. Please check connection.")
                ret, frame = cap.read()
                im_in = np.array(frame)
            # Load the demo image
            else:
                im_file = os.path.join(args.image_dir, imglist[num_images])
                im_in = cv2.imread(im_file)
            # bgr
            im = im_in

            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_()

                # pdb.set_trace()
            det_tic = time.time()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extract predicted params
            contact_vector = loss_list[0][0]  # hand contact state info
            offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

            # get hand contact
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im2show = np.copy(im)
            obj_dets, hand_dets = None, None
            for j in xrange(1, len(pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                elif pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()
            if (obj_dets is not None) and (hand_dets is not None):
                hand_to_obj_index_match = filter_object(obj_dets, hand_dets)
            else:
                hand_to_obj_index_match = None
            obj_dets_info = None
            if obj_dets is not None:
                obj_dets_info = get_det_info(obj_dets, det_type='objs')
            hand_dets_info = None
            if hand_dets is not None:
                hand_dets_info = get_det_info(hand_dets, det_type='hands')
            # Export hand and object detection information to a json file.
            frame_id = imglist[num_images]
            export_data = {frame_id: {'object_detections': obj_dets_info,
                                      'hand_detections': hand_dets_info,
                                      'hand_to_object_match': hand_to_obj_index_match}
                           }
            data_to_export.append(export_data)
            if vis:
                # visualization
                im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj,
                                                              font_path=font_path)

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            if webcam_num == -1:
                sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                                 .format(num_images + 1, len(imglist), detect_time, nms_time))
                sys.stdout.flush()

            if vis and webcam_num == -1:
                save_vis_filename = os.path.basename(imglist[num_images]).split(sep='.')[0] + '.png'
                result_path = os.path.join(save_imgs_dir, save_vis_filename)
                im2show.save(result_path)  # final image is saved to disk here
            else:
                im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                cv2.imshow("frame", im2showRGB)
                total_toc = time.time()
                total_time = total_toc - total_tic
                frame_rate = 1 / total_time
                print('Frame rate:', frame_rate)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        with open(save_json_filepath, mode='w') as f:
            json.dump(data_to_export, f, indent=4)
        if webcam_num >= 0:
            cap.release()
            cv2.destroyAllWindows()
