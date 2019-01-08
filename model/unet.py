import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

from giana.helpers.utils import dice_score, make_one_hot


class DoubleConv(nn.Module):
    '''
    perform (conv => BatchNorm => ReLU) * 2
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.conv(x)
        return x
   
class DownConv(nn.Module):
    '''
    perform (maxpool => DoubleConv)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownConv, self).__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2),
                                     DoubleConv(in_channels, out_channels, 
                                                kernel_size, stride, padding))

    def forward(self, x):
        x = self.mp_conv(x)
        return x

class UpConv(nn.Module):
    '''
    perform 
    (either upsample using bilinear interpolation) 
    OR
    (learn the parameters for upsampling using transposed convolution)
    '''
    def __init__(self, in_channels, out_channels, mode='bilinear',
                 kernel_size=3, stride=2, padding=1):
        super(UpConv, self).__init__()
        self.mode = mode
        if mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            ## Dont understand this part
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size, stride, padding)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_copied):
        if self.mode == 'bilinear':
            x = self.up(x)
        else:
            x = self.up(x, output_size=x_copied.shape)
        diff_h = x.shape[2] - x_copied.shape[2]
        diff_w = x.shape[3] - x_copied.shape[3]

        x_copied = F.pad(x_copied, (diff_w//2, int(np.ceil(diff_w/2)), 
                                    diff_h//2, int(np.ceil(diff_h/2))))
                                
        x = torch.cat([x_copied, x], 1)
        x = self.conv(x)
        return x


class UpConvV2(nn.Module):
    '''
    perform 
    (upsample using bilinear interpolation) 
    
    '''
    def __init__(self, in_channels, out_channels, mode='bilinear', scale_factor=2):
        super(UpConvV2, self).__init__()
        self.mode = mode
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_copied=None):
        if x_copied is not None:
            diff_h = x.shape[2] - x_copied.shape[2]
            diff_w = x.shape[3] - x_copied.shape[3]

            x_copied = F.pad(x_copied, (diff_w//2, int(np.ceil(diff_w/2)), 
                                        diff_h//2, int(np.ceil(diff_h/2))))
            x = torch.cat([x_copied, x], 1)

        
        x = self.up(x)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(in_channels=n_channels, out_channels=64)
        self.down_1 = DownConv(in_channels=64, out_channels=128)
        self.down_2 = DownConv(in_channels=128, out_channels=256)
        self.down_3 = DownConv(in_channels=256, out_channels=512)
        self.down_4 = DownConv(in_channels=512, out_channels=512)

        self.up_1 = UpConv(in_channels=1024, out_channels=256, mode='Transpose2D')
        self.up_2 = UpConv(in_channels=512, out_channels=128, mode='Transpose2D')
        self.up_3 = UpConv(in_channels=256, out_channels=64, mode='Transpose2D')
        self.up_4 = UpConv(in_channels=128, out_channels=64, mode='Transpose2D')
        self.out_conv = DoubleConv(in_channels=64, out_channels=n_classes)

    def forward(self, x):
        # print('x = ', x.shape)
        x_1 = self.in_conv(x)
        # print('x_1 = ', x_1.shape)
        x_2 = self.down_1(x_1)
        # print('x_2 = ', x_2.shape)
        x_3 = self.down_2(x_2)
        # print('x_3 = ', x_3.shape)
        x_4 = self.down_3(x_3)
        # print('x_4 = ', x_4.shape)
        x_5 = self.down_4(x_4)
        # print('x_5 = ', x_5.shape)
        x = self.up_1(x_5, x_4)
        # print('x_6 = ', x.shape)
        x = self.up_2(x, x_3)
        # print('x_7 = ', x.shape)
        x = self.up_3(x, x_2)
        # print('x_8 = ', x.shape)
        x = self.up_4(x, x_1)
        # print('x_9 = ', x.shape)
        x = self.out_conv(x)
        # print('x_10 = ', x.shape)


        return x


    def predict(self, test_loader, score_type, max_iterations=30, viz=None):
        correct = 0
        total = 0
        first = True
        score = 0
        idx = 0
        self.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                # print(x.shape, y.shape)
                y_onehot = make_one_hot(y, 2)
                if torch.cuda.is_available():
                    x = x.cuda()
                    y_onehot = y_onehot.cuda()

                predicted = self.forward(x)
                predicted_softmax = F.softmax(predicted, dim=1)

            
                if score_type == 'accuracy':
                    score += self.calc_acc(predicted, y)
                elif score_type == 'dice':
                    score += self.calc_dice_score(predicted_softmax, y_onehot)

                if viz is not None:
                    image_args = {'normalize':True, 'range':(0, 1)}
                    # viz.show_image_grid(images=x.cpu()[:, 0, ].unsqueeze(1), name='Images_test')
                    viz.show_image_grid(images=y, name='TestGT')
                    viz.show_image_grid(images=predicted_softmax.cpu()[:, 0, ].unsqueeze(1), name='TestPred_1', image_args=image_args)
                    viz.show_image_grid(images=predicted_softmax.cpu()[:, 1, ].unsqueeze(1), name='TestPred_2', image_args=image_args)


                if idx==max_iterations:
                    break

            # print('Iteration: {}, Score: {}'.format(idx, score))

        return score/idx

    def calc_acc(self, predictions, targets):
        correct = (predictions == targets).sum()
        accuracy = np.round(100.*correct/total, 3)
        return accuracy

    def calc_dice_score(self, predictions, targets):
        return dice_score(predictions, targets)

class ResUNet(nn.Module):
    def __init__(self, n_classes):
        super(ResUNet, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        encoder = nn.Sequential(*list(resnet.children())[0:6])
        # for params in encoder.parameters():
        #     params.requires_grad = False

        self.in_conv = encoder[0:4]
        self.down_1 = encoder[4]
        self.down_2 = encoder[5]
        # self.down_3 = encoder[6]
        self.down_4 = DownConv(in_channels=512, out_channels=512)

        self.up_1 = UpConv(in_channels=1024, out_channels=256, mode='Transpose2D')
        self.up_2 = UpConv(in_channels=512, out_channels=64, mode='Transpose2D')
        # self.up_3 = UpConv(in_channels=128, out_channels=32, mode='bilinear')
        # self.up_4 = UpConv(in_channels=64, out_channels=32, mode='bilinear')
        self.up_3 = UpConvV2(in_channels=128, out_channels=32, scale_factor=4)
        # self.up_4 = UpConv(in_channels=128, out_channels=64, mode='Transpose2D')
        self.out_conv = DoubleConv(in_channels=32, out_channels=n_classes)

    def forward(self, x):
        # print('x = ', x.shape)
        x_1 = self.in_conv(x)
        # print('x_1 = ', x_1.shape)
        x_2 = self.down_1(x_1)
        # print('x_2 = ', x_2.shape)
        x_3 = self.down_2(x_2)
        # print('x_3 = ', x_3.shape)
        # x_4 = self.down_3(x_3)
        # # print('x_4 = ', x_4.shape)
        x_4 = self.down_4(x_3)
        # print('x_4 = ', x_4.shape)
        
        x = self.up_1(x_4, x_3)
        # print('x_5 = ', x.shape)
        x = self.up_2(x, x_2)
        # print('x_6 = ', x.shape)
        x = self.up_3(x, x_1)
        # print('x_7 = ', x.shape)
        x = self.out_conv(x)
        # print('x_8 = ', x.shape)


        return x


    def predict(self, test_loader, score_type, max_iterations=30, viz=None):
        correct = 0
        total = 0
        first = True
        score = 0
        idx = 0
        self.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                # print(x.shape, y.shape)
                y_onehot = make_one_hot(y, 2)
                if torch.cuda.is_available():
                    x = x.cuda()
                    y_onehot = y_onehot.cuda()

                predicted = self.forward(x)
                predicted_softmax = F.softmax(predicted, dim=1)

            
                if score_type == 'accuracy':
                    score += self.calc_acc(predicted, y)
                elif score_type == 'dice':
                    score += self.calc_dice_score(predicted_softmax, y_onehot)

                if viz is not None:
                    image_args = {'normalize':True, 'range':(0, 1)}
                    # viz.show_image_grid(images=x.cpu()[:, 0, ].unsqueeze(1), name='Images_test')
                    viz.show_image_grid(images=y, name='TestGT')
                    viz.show_image_grid(images=predicted_softmax.cpu()[:, 0, ].unsqueeze(1), name='TestPred_1', image_args=image_args)
                    viz.show_image_grid(images=predicted_softmax.cpu()[:, 1, ].unsqueeze(1), name='TestPred_2', image_args=image_args)


                if idx==max_iterations:
                    break

            # print('Iteration: {}, Score: {}'.format(idx, score))

        return score/idx

    def calc_acc(self, predictions, targets):
        correct = (predictions == targets).sum()
        accuracy = np.round(100.*correct/total, 3)
        return accuracy

    def calc_dice_score(self, predictions, targets):
        return dice_score(predictions, targets)



def main():
    input_image = torch.empty(3, 3, 640, 640)
    unet = UNet(3, 2)
    output_image = unet(input_image)
    # print(output_image.shape)

    # nn.CrossEntropyLoss





