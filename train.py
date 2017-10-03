import sys
import torch
# import visdom
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
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *

import time
import os

def train(args):
    # time start
    start = 0

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    train_loader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)


    # Setup visdom for visualization
    # vis = visdom.Visdom()

    # loss_window = vis.line(X=torch.zeros((1,)).cpu().numpy(),
    #                        Y=torch.zeros((1)).cpu().numpy(),
    #                        opts=dict(xlabel='minibatches',
    #                                  ylabel='Loss',
    #                                  title='Training Loss',
    #                                  legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)

    if torch.cuda.is_available():
        model.cuda(args.cuda_index)
    # if torch.cuda.is_available():
    #     model.cuda(args.cuda_index)
    #     test_image, test_segmap = loader[0]
    #     test_image = Variable(test_image.unsqueeze(0).cuda(args.cuda_index))
    # else:
    #     test_image, test_segmap = loader[0]
    #     test_image = Variable(test_image.unsqueeze(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    val_loader_instance = data_loader(data_path, split='val', is_transform=True, img_size=(args.img_rows, args.img_cols))
    val_loader = data.DataLoader(val_loader_instance, batch_size=args.batch_size, num_workers=4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    loss_arr = [-1]*len(train_loader)*(args.n_epoch+1)
    print("Train data size: ", len(train_loader)*args.batch_size)
    for epoch in range(args.start_epoch, args.n_epoch):
        for i, (images, labels) in enumerate(train_loader):
            #print('iteration: {}'.format(i))
            if torch.cuda.is_available():
                images = Variable(images.cuda(args.cuda_index))
                labels = Variable(labels.cuda(args.cuda_index))
            else:
                images = Variable(images)
                labels = Variable(labels)

            iter = len(train_loader)*epoch + i
            poly_lr_scheduler(optimizer, args.l_rate, iter, power=0) # power = 0 to disable scheduler 
            
            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            # vis.line(
            #     X=torch.ones((1, 1)).cpu().numpy() * i,
            #     Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu().numpy(),
            #     win=loss_window,
            #     update='append')

            loss_arr[i] = loss.data[0]
            if (i+1) % 20 == 0:
                #print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))
                end = time.time()
                print("Epoch [%d/%d] Iteration [%d/%d] Loss: %.4f time(sec): %.1f" % (epoch+1, args.n_epoch, i, len(train_loader), loss.data[0], end-start))
                start = time.time()

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
        # GCP storage!
        #if (epoch+1)%2 == 0:
            #torch.save(model, "{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))
        if (epoch+1)%args.validate_every == 0:
            print("Validation starting on epoch: ", epoch)
            validate(train_loader, model, n_classes)
            validate(val_loader, model, n_classes)
            # filename_prefix = args.arch+ '_' + str(args.batch_size) + '_' + str(arg.l_rate)
            # save_checkpoint({
            #             'epoch': epoch + 1,
            #             'arch': args.arch,
            #             'state_dict': model.state_dict(),
            #             #'best_prec1': best_prec1,
            #             'optimizer' : optimizer.state_dict(),
            #         }, loss_arr, False, epoch, filename_prefix, 2)

def save_checkpoint(state, loss_arr, is_best, epoch, filename_prefix, max_to_keep=3):
    model_filename_prefix = filename_prefix + '_model_'
    model_filename_suffix = '.pth.tar'
    final_model_filename = model_filename_prefix + str(epoch) + model_filename_suffix
    torch.save(state, final_model_filename)
    clean_exceeding_files(model_filename_prefix, max_to_keep)

    loss_filename_prefix = filename_prefix + '_loss_array_'
    loss_filename_suffix = '.npy'
    final_loss_filename = loss_filename_prefix + str(epoch) + loss_filename_suffix
    np.save(final_loss_filename, loss_arr)
    clean_exceeding_files(loss_filename_prefix, 1)

    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')

# assumes filename is like this $PREFIX$EPOCH.$SUFFIX
# assumes current directory only
def clean_exceeding_files(filename_prefix, max_to_keep):
    directory_filenames = os.listdir('./')
    matched_filenames = []
    for filename in directory_filenames:
        if filename.startswith(filename_prefix):
            matched_filenames.append(filename)
    matched_filenames = sorted(matched_filenames, key=lambda filename: int(filename[ len(filename_prefix): filename.index('.')]) )
    if len(matched_filenames) > max_to_keep:
        delete_filenames = matched_filenames[:-max_to_keep]
        for filename in delete_filenames:
            os.remove(filename)

def validate(val_loader, model, n_classes):
    # switch to evaluate mode
    model.eval()

    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(val_loader)):
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

    # UNCOMMENT IF YOU WANT PER CLASS IOU
    # for i in range(n_classes):
    #     print i, class_iou[i]     

    # switch to train mode
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--cuda_index', default=0, type=int, metavar='N',
                    help='gpu index')
    parser.add_argument('--validate_every', default=5, type=int, metavar='N',
                    help='validate every x epochs')

    args = parser.parse_args()
    print("Training arch {} dataset {} batchsize {} size {}x{} cuda index {}".format(args.arch, args.dataset, args.batch_size, args.img_rows, args.img_cols, args.cuda_index))
    train(args)
