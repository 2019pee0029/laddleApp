import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate, ReadImagesFromFolder,AlignCollateTest
from .model import RecoModel
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = RecoModel(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor(
                [opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    # prune after "end of sentence" token ([s])
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(
                    f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
#     parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
#     parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
#     parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
#     """ Data processing """
#     parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
#     parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
#     parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
#     parser.add_argument('--rgb', action='store_true', help='use rgb input')
#     parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
#     parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
#     parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
#     """ Model Architecture """
#     parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
#     parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
#     parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
#     parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
#     parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
#     parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
#     parser.add_argument('--output_channel', type=int, default=512,
#                         help='the number of output channel of Feature extractor')
#     parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

#     opt = parser.parse_args()

#     """ vocab / character number configuration """
#     if opt.sensitive:
#         opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

#     cudnn.benchmark = True
#     cudnn.deterministic = True
#     opt.num_gpu = torch.cuda.device_count()

# new code


class Parser():
    def __init__(self):
        self.batch_size = None
        self.image_folder = None
        self.saved_model = None
        self.batch_max_length = None
        self.imgH = None
        self.imgW = None
        self.rgb = None
        self.character = None
        self.sensitive = None
        self.PAD = None
        self.Transformation = None
        self.FeatureExtraction = None
        self.SequenceModeling = None
        self.Prediction = None
        self.workers = None
        self.num_fiducial = None
        self.input_channel = None
        self.output_channel = None
        self.hidden_size = None

    def add_argument(self, name=None, type=None, help='help', required=False, default=None, action=None):
        if default:
            setattr(self, name[2:], default)


def initializeParsers():
    parser = Parser()
    parser.add_argument('--image_folder', required=True,
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True,
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser

    """ vocab / character number configuration """
    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    return opt


def LoadRecoModel(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = RecoModel(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    print('loaded model')
    model.eval()
    return model

import time
def PredictReco(opt, model):
    t4 = time.time()

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    
    #preds
    t5 = time.time()
    print(f'time elapsed inside reco t5 {t5-t4}')
    res = ''
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            t6 = time.time()
            print(f'time elapsed inside reco t6 {t6-t5}')
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor(
                [opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            t7 = time.time()
            print(f'time elapsed inside reco t7 {t7-t6}')

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                t8 = time.time()
                print(f'time elapsed inside reco t8 {t8-t7}')
                preds = model(image, text_for_pred, is_train=False)
                t9 = time.time()
                print(f'time elapsed inside reco t9 {t9-t8}')

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            #log = open(f'./log_demo_result.txt', 'a')
            log = open(f'./Laddle-Id-Detection.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    # prune after "end of sentence" token ([s])
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
            res = [img_name, confidence_score, pred]
    t10 = time.time()
    print(f'time elapsed inside reco t10 {t10-t9}')
    return res


def PredictRecog(opt, model,image):
    # read the image from result folder
    if len(image) < 0:
        return 'not detected'
    try:
        img = Image.fromarray(image).convert('L')
    except Exception as e:
        print(e, image)
        return '0'
    #img=Image.open(image).convert('L')
    
    AlignCollate_demo = AlignCollateTest(
        imgH=100, imgW=32, keep_ratio_with_pad=opt.PAD)

    image_tensors = AlignCollate_demo(img)
    res = ''
    converter = AttnLabelConverter(opt.character)
    with torch.no_grad():
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor(
            [opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(
            batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        #log = open(f'./log_demo_result.txt', 'a')
        log = open(f'./Laddle-Id-Detection.txt', 'a')       #mine
        dashed_line = '-' * 80
        head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        image_path_list = [opt.test_image]
        for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                # prune after "end of sentence" token ([s])
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
            log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')  #mine

        res = [img_name, confidence_score, pred]
    return res
