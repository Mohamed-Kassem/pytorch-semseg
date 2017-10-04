import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
# FCN32s
class fcn32s(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        out = F.upsample_bilinear(score, x.size()[2:])

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print type(l1), dir(l1),
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

class fcn16s(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        out = F.upsample_bilinear(score, x.size()[2:])

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

# FCN 8s
class fcn8s(nn.Module):

    def __init__(self, n_classes=21, kassem=False, exp_index=0, learned_billinear=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.kassem = kassem
        self.exp_index = exp_index

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        # Kassem edges addition - start
        if self.kassem:
            # self.get_edges = nn.Conv2d(3, 2, 3, padding=1, bias=False)

            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         # print(m)
            #         kernel_size = m.weight.data.size()
            #         # print(kernel_size)
            #         if kernel_size[0] == 2 and kernel_size[1] == 3:
            #             print('NOTE THAT THIS SHOULD BE PRINTED ONCE')
            #             #print(kernel_size)
            #             #print(m.weight.data)
            #             m.weight.data.numpy()[0,:,:,:] = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            #             m.weight.data.numpy()[1,:,:,:] = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            if self.exp_index == 0:
                self.get_edges = nn.Conv2d(3, 4, 5, padding=3, bias=False)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        # print(m)
                        kernel_size = m.weight.data.size()
                        # print(kernel_size)
                        if kernel_size[0] == 4 and kernel_size[1] == 3:
                            print('NOTE THAT THIS SHOULD BE PRINTED ONCE')
                            #print(kernel_size)
                            #print(m.weight.data)
                            m.weight.data.numpy()[0,:,:,:] = np.array([
                                [0,0,0,0,0],
                                [0,-1,0,1,0],
                                [0,-2,0,2,0],
                                [0,-1,0,1,0],
                                [0,0,0,0,0]]) / 2
                            m.weight.data.numpy()[1,:,:,:] = np.array([
                                [0,0,0,0,0],
                                [0,-1,0,1,0],
                                [0,-2,0,2,0],
                                [0,-1,0,1,0],
                                [0,0,0,0,0]]).T / 2
                            m.weight.data.numpy()[2,:,:,:] = np.array([
                                [-2/8,-1/5,0,1/5,2/8],
                                [-2/5,-1/2,0,1/2,2/5],
                                [-2/4,-1/1,0,1/1,2/4],
                                [-2/5,-1/2,0,1/2,2/5],
                                [-2/8,-1/5,0,1/5,2/8]])
                            m.weight.data.numpy()[3,:,:,:] = np.array([
                                [-2/8,-1/5,0,1/5,2/8],
                                [-2/5,-1/2,0,1/2,2/5],
                                [-2/4,-1/1,0,1/1,2/4],
                                [-2/5,-1/2,0,1/2,2/5],
                                [-2/8,-1/5,0,1/5,2/8]]).T
            elif self.exp_index == 1:
                self.get_edges = nn.Conv2d(3, 6, 7, padding=5, bias=False)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        # print(m)
                        kernel_size = m.weight.data.size()
                        # print(kernel_size)
                        if kernel_size[0] == 6 and kernel_size[1] == 3:
                            print('NOTE THAT THIS SHOULD BE PRINTED ONCE')
                            #print(kernel_size)
                            #print(m.weight.data)
                            m.weight.data.numpy()[0,:,:,:] = np.array([
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0],
                                [0,0,-1,0,1,0,0],
                                [0,0,-2,0,2,0,0],
                                [0,0,-1,0,1,0,0],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0]]) / 2
                            m.weight.data.numpy()[1,:,:,:] = np.array([
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0],
                                [0,0,-1,0,1,0,0],
                                [0,0,-2,0,2,0,0],
                                [0,0,-1,0,1,0,0],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0]]).T / 2
                            m.weight.data.numpy()[2,:,:,:] = np.array([
                                [0,0,0,0,0,0,0],
                                [0,-2/8,-1/5,0,1/5,2/8,0],
                                [0,-2/5,-1/2,0,1/2,2/5,0],
                                [0,-2/4,-1/1,0,1/1,2/4,0],
                                [0,-2/5,-1/2,0,1/2,2/5,0],
                                [0,-2/8,-1/5,0,1/5,2/8,0],
                                [0,0,0,0,0,0,0]])
                            m.weight.data.numpy()[3,:,:,:] = np.array([
                                [0,0,0,0,0,0,0],
                                [0,-1,-2,0,2,1,0],
                                [0,-4,-8,0,8,4,0],
                                [0,-6,-12,0,12,6,0],
                                [0,-4,-8,0,8,4,0],
                                [0,-1,-2,0,2,1,0],
                                [0,0,0,0,0,0,0]]).T
                            m.weight.data.numpy()[4,:,:,:] = np.array([
                                [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18],
                                [-3/13, -2/8 , -1/5 , 0,  1/5 , 2/8 , 3/13],
                                [-3/10, -2/5 , -1/2 , 0,  1/2 , 2/5 , 3/10],
                                [-3/9 , -2/4 , -1/1 , 0,  1/1 , 2/4 , 3/9 ],
                                [-3/10, -2/5 , -1/2 , 0,  1/2 , 2/5 , 3/10],
                                [-3/13, -2/8 , -1/5 , 0,  1/5 , 2/8 , 3/13],
                                [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18]])
                            m.weight.data.numpy()[5,:,:,:] = np.array([
                                [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18],
                                [-3/13, -2/8 , -1/5 , 0,  1/5 , 2/8 , 3/13],
                                [-3/10, -2/5 , -1/2 , 0,  1/2 , 2/5 , 3/10],
                                [-3/9 , -2/4 , -1/1 , 0,  1/1 , 2/4 , 3/9 ],
                                [-3/10, -2/5 , -1/2 , 0,  1/2 , 2/5 , 3/10],
                                [-3/13, -2/8 , -1/5 , 0,  1/5 , 2/8 , 3/13],
                                [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18]]).T

            # kassem adding edges- THREE OF THIS
            # self.edges_conv = nn.Conv2d(2, self.n_classes, 3, padding=1, bias=True) # 10 should be classes num

            # kassem concatenating edges- THREE OF THIS'
            if self.exp_index == 0:
                self.out_conv = nn.Conv2d(self.n_classes+4, self.n_classes, 3, padding=1, bias=True) # 10 should be classes num
            elif self.exp_index == 0:
                self.out_conv = nn.Conv2d(self.n_classes+6, self.n_classes, 3, padding=1, bias=True) # 10 should be classes num

            # print('Model for loop init finished')
        # print('Model init finished')
        # Kassem edges addition - end

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        # print('Model forward start')
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample_bilinear(score, score_pool3.size()[2:])
        score += score_pool3
        if not self.kassem:
            out = F.upsample_bilinear(score, x.size()[2:])
            # print('out')
            # print(out[0, 0, 50:53, 50:53])

        # Kassem edges addition - start
        if self.kassem:
            y = self.get_edges(x)
            y_squared = y*y
            if self.exp_index == 0:
                edges_mag_1 = torch.sqrt(y_squared[:, 0:1, :, :] + y_squared[:, 1:2, :, :]) # using 0:1 AND 1: to keepDim = 4
                edges_dir_1 = torch.atan(y[:, 1:2, :, :]/ (y[:, 0:1, :, :] + 1e-5 ) )
                edges_mag_2 = torch.sqrt(y_squared[:, 2:3, :, :] + y_squared[:, 3:4, :, :])
                edges_dir_2 = torch.atan(y[:, 3:4, :, :]/ (y[:, 1:2, :, :] + 1e-5 ) )
                edges_cat = torch.cat((edges_mag_1, edges_dir_1, edges_mag_2, edges_dir_2), 1)

            elif self.exp_index == 1:
                edges_mag_1 = torch.sqrt(y_squared[:, 0:1, :, :] + y_squared[:, 1:2, :, :]) # using 0:1 AND 1: to keepDim = 4
                edges_dir_1 = torch.atan(y[:, 1:2, :, :]/ (y[:, 0:1, :, :] + 1e-5 ) )
                edges_mag_2 = torch.sqrt(y_squared[:, 2:3, :, :] + y_squared[:, 3:4, :, :])
                edges_dir_2 = torch.atan(y[:, 3:4, :, :]/ (y[:, 1:2, :, :] + 1e-5 ) )
                edges_mag_3 = torch.sqrt(y_squared[:, 4:5, :, :] + y_squared[:, 5:6, :, :])
                edges_dir_3 = torch.atan(y[:, 5:6, :, :]/ (y[:, 4:5, :, :] + 1e-5 ) )
                edges_cat = torch.cat((edges_mag_1, edges_dir_1, edges_mag_2, edges_dir_2, edges_mag_3, edges_dir_3), 1)

            # kassem adding edges - THREE OF THIS
            # edges_conv = self.edges_conv(edges_cat)
            # out = F.upsample_bilinear(score, x.size()[2:]) + edges_conv

            # kassem concatenating edges- THREE OF THIS
            out_1 = F.upsample_bilinear(score, x.size()[2:])
            out_2 = torch.cat( (out_1, edges_cat), 1)
            out = self.out_conv(out_2)


            # print('out')
            # print(out[0, 0, 50:53, 50:53])
        # Kassem edges addition - end

        # print('Model forward finished')

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]