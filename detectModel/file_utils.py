# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
from recoModel.demo import *
import logging
log = logging.getLogger('Laddle-Id-detection')

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

import random
import datetime

dict_boxes ={}
def saveResult(img_file, img, boxes, opt, reco, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        direction = 0
        img = np.array(img)
        #dirname = datetime.date

        # make result file list
        #filename, file_ext = os.path.splitext(os.path.basename(img_file))
        filename = str(random.randint(0,10000))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        detexList = []
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            # f.write(strResult)
                
            poly = poly.reshape(-1, 2)
            # print("poly is {}".format(poly))
            x1 = int(poly[0][0])
            y1 = int(poly[0][1])
            x2 = int(poly[1][0])
            y2 = int(poly[2][1])
            #print("x1,x2,y1,y2",x1,x2,y1,y2)
            img_name = dirname + "res_" + filename + '_{}_'.format(i) + '.jpg'
            detexList.append(img_name)
            cropped = img[y1:y2,x1:x2]
            if cropped.size > 0:
                imgname,confidence_,texts = PredictRecog(opt,reco,cropped)
                img_name = dirname + texts + '_' + filename + '.jpg'
                #proc = cv2.resize(cropped, (32, 32))
                position = '' 
                if confidence_>0.90 and len(texts)==2 and texts.isdigit():
                    cv2.imwrite(img_name, cropped)
                    position = np.mean([x1,x2]) # centre position of the detected text
                        # check if last position if greater than current then direction = 1 else direction = -1
                    if texts in dict_boxes.keys():
                        if dict_boxes[texts]>position:
                            direction = 1
                        else:
                            direction = -1
                    dict_boxes[texts] = position
                    log.info(f'the position of detected text {texts} is {position} and direction is {direction}')

            cv2.polylines(img, [poly.reshape((-1, 1, 2))],
                          True, color=(0, 0, 255), thickness=1)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts+"_" + str(direction)), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts + "_" + str(direction)), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
        return detexList,img

def saveResultFrame(imgfile,img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        #img = np.array(img)

        # make result file list
        #filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        filename = 'test'
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        detexList = []
        BoxList =[]
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            # f.write(strResult)
                
            poly = poly.reshape(-1, 2)
            #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            # print("poly is {}".format(poly))
            x1 = int(poly[0][0])
            y1 = int(poly[0][1])
            x2 = int(poly[1][0])
            y2 = int(poly[2][1])
            #print("x1,x2,y1,y2",x1,x2,y1,y2)
            img_name = dirname + "res_" + filename + '_{}_'.format(i) + '.jpg'
            detexList.append(img_name)
            cropped = img[y1:y2,x1:x2]

            if cropped.size>0:
                #cv2.imwrite(img_name,cropped)
                BoxList.append(cropped)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
        
        return detexList, BoxList
