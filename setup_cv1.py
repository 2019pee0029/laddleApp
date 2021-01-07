import sys
sys.path.append('D://laddleApp')
from recoModel.demo import *
from recoModel.model import RecoModel
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data



from recoModel.utils import CTCLabelConverter, AttnLabelConverter
from recoModel.dataset import RawDataset, AlignCollate
from detectModel.test import *
from PIL import Image
import cv2 as cv
import threading
from log import log, setupTimeRotatedLog

setupTimeRotatedLog("Laddle-Id-detection.log", log)

#recognizer model
opt = initializeParsers()
opt.FeatureExtraction = 'ResNet'
opt.SequenceModeling = 'BiLSTM'
opt.Prediction = 'Attn'
opt.test_image = 'street_name.jpg'
opt.image_folder = 'result'
opt.Transformation = 'TPS'
opt.saved_model = 'D:\\laddleApp\\recoModel\\weights\\TPS-ResNet-BiLSTM-Attn.pth'
# detector model
args = initializeDetectModelParsers()
args.trained_model = "D:\\laddleApp\\detectModel\\weights\\craft_mlt_25k.pth"
args.test_folder = "D:\\laddleApp\\detectModel\\test2\\"
net = LoadDetectionModel(args)
reco = LoadRecoModel(opt)
#res = Recognize(opt,reco)
log.info("loaded all models")



class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(
            'rtsp://150.0.147.105/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')  #172.24.136.242:554
        #self.video = cv.VideoCapture('D:\\AI DS Projects\\rec_17_30.avi')
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.grabbed, self.frame
        return ret,image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def DetectTextArea(image):
    "takes an image path and returns detected boxes"
    preds = PredictDetection(args, net, image, opt,reco)
    
    return preds

cam = VideoCamera()
ret, frame = cam.get_frame()

while(ret):
    ret, frame = cam.get_frame()
    preds,img = (DetectTextArea(frame))
    cv.imshow('frame',img)
    cv.moveWindow('frame', 200, 200)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break

