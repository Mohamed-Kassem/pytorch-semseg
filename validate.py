import sys
import torch
#import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

def validate(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4)

    # Setup Model
    checkpoint = torch.load(args.model_path)
    model = get_model(args.arch, n_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda(args.cuda_index)

    gts, preds = [], []
    print("Validation data size: ", len(valloader)*args.batch_size)
    for i, (images, labels) in tqdm(enumerate(valloader)):
        if torch.cuda.is_available():
            images = Variable(images.cuda(args.cuda_index))
            labels = Variable(labels.cuda(args.cuda_index))
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)
        # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    score, class_iou = scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print k, v

    for i in range(n_classes):
        print i, class_iou[i] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val', 
                        help='Split of dataset to test on')
    parser.add_argument('--cuda_index', default=0, type=int, metavar='N',
                    help='Specify gpu index')
    args = parser.parse_args()
    print("Validating arch {} dataset {} batchsize {} size {}x{} cuda index {}".format(args.arch, args.dataset, args.batch_size, args.img_rows, args.cols, args.cuda_index))
    validate(args)
