"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import sys
sys.path.append('D://PyOpenCVcamera//detectModel')
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT
import logging

log = logging.getLogger('Laddle-Id-detection')

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")





class Parser():
    def __init__(self):
        self.text_threshold=None
        self.low_text=None
        self.link_threshold=None
        self.cuda=None
        self.canvas_size=None
        self.mag_ratio=None
        self.show_time=None
        self.trained_model=None
        self.test_folder=None
    def add_argument(self, name=None, type=None, help='help', required=False, default=None, action=None):
        if default:
            setattr(self, name[2:], default)
            
def initializeDetectModelParsers():
    parser = Parser()
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    log.info('created')
    return parser

def test_net(net, image, text_threshold, link_threshold, low_text, cuda,args):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    x = x.contiguous()
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        #y, feature = net(x)
        y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text



# if __name__ == '__main__':
#     # load net
#     net = CRAFT()     # initialize

#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     net.load_state_dict(copyStateDict(torch.load(args.trained_model,map_location='cpu')))

#     if args.cuda:
#         net = net.cuda()
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = False

#     net.eval()

#     t = time.time()

#     # load data
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
#         image = imgproc.loadImage(image_path)

#         bboxes, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda)

#         # save score text
#         filename, file_ext = os.path.splitext(os.path.basename(image_path))
#         mask_file = result_folder + "/res_" + filename + '_mask.jpg'
#         cv2.imwrite(mask_file, score_text)

#         file_utils.saveResult(image_path, image[:,:,::-1], bboxes, dirname=result_folder)

#     print("elapsed time : {}s".format(time.time() - t))

def LoadDetectionModel(args):
    net = CRAFT()     # initialize
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    net.load_state_dict(copyStateDict(torch.load(args.trained_model)))#,map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    return net


def PredictDetection(args, net, image_path, opt, reco):
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    t = time.time()
    # load data
    # for k, image_path in enumerate(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)
    bboxes, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda,args)

    # save score text
    #filename, file_ext = os.path.splitext(os.path.basename(image_path))
    #mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    #cv2.imwrite(mask_file, score_text)

    fl = file_utils.saveResult(image_path, image[:, :, ::-1], bboxes, opt, reco, dirname=result_folder)

    print("elapsed time detecting : {}s".format(time.time() - t))
    log.info(f'elapsed time detecting : {time.time() - t}s')
    return fl

def PredictDetectionFrame(args,net,image_path):
    """ For test images in a folder """

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    t = time.time()
    # load data
    # for k, image_path in enumerate(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)

    bboxes, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda,args)
    #print(f'boxes sare {bboxes}')
    # save score text
    filename, file_ext = 'test','.jpg'
    #mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    #cv2.imwrite(mask_file, score_text)
    result_folder

    fl = file_utils.saveResultFrame(image_path, image[:,:,::-1], bboxes, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
    return fl
