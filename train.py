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

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *

import time
import sys

def train(args):
    # time start
    start = 0

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

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
        model.cuda(0)
        test_image, test_segmap = loader[0]
        test_image = Variable(test_image.unsqueeze(0).cuda(0))
    else:
        test_image, test_segmap = loader[0]
        test_image = Variable(test_image.unsqueeze(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    loss_arr = [-1]*len(trainloader)*(args.n_epoch+1)
    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)

            iter = len(trainloader)*epoch + i
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
                print("Epoch [%d/%d] Iteration [%d/%d] Loss: %.4f time(sec): %.1f" % (epoch+1, args.n_epoch, i, len(trainloader), loss.data[0], end-start))
                start = time.time()

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
        np.save("loss_array_epoch_{}.npy".format(epoch), loss_arr)
        # GCP storage!
        if (epoch+1)%2 == 0:
            torch.save(model, "{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))

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
    args = parser.parse_args()

    f = open("logs", 'w')
    sys.stdout = f
    train(args)
    f.close()
