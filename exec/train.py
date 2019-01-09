import sys, os
# sys.path.append('/home/avemuri/DEV/projects/endovis2018-challenge/')
sys.path.append('/media/anant/dev/src/GIANA/')


from dataloader.giana_dataloader import giana_data_pipeline
from helpers.utils import SoftDiceLoss, make_one_hot, dice_score
from model.unet import UNet, ResUNet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from trixi.logger import PytorchVisdomLogger

EPOCHS = 150
MAX_ITERATIONS = 200
PRINT_AFTER_ITERATIONS = 10
LEARNING_RATE = 0.001
SCORE_TYPE = 'dice'
MAX_VALIDATION_ITERATIONS = 55


def train(datafile):

    # model = ResUNet(n_classes=2)
    model = UNet(n_channels=3, n_classes=2)

    if torch.cuda.is_available():
        model.cuda()
    # criterion = SoftDiceLoss(batch_dice=True)
    criterion_CE = nn.CrossEntropyLoss()
    criterion_SD = SoftDiceLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum = 0.9)

    vis = PytorchVisdomLogger(name="GIANA", port=8080)

    giana_transform, giana_train_loader, giana_valid_loader = giana_data_pipeline(datafile)
    
    for epoch in range(EPOCHS):
        iteration = 0
        for iteration, (images, labels) in enumerate(giana_train_loader):
            # print('TRAIN', images.shape, labels.shape)
            images, labels = giana_transform.apply_transform([images, labels])
            
            labels_onehot = make_one_hot(labels, 2)
        # for images, labels in giana_pool.imap_unordered(giana_transform.apply_transform, giana_iter):  
            if torch.cuda.is_available():
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()

            optimizer.zero_grad()
            model.train()
            predictions = model(images)
            predictions_softmax = F.softmax(predictions, dim=1)
            
            # loss = 0.75 * criterion_CE(predictions, labels.squeeze().cuda().long()) + 0.25 * criterion_SD(predictions_softmax, labels_onehot)
            loss = criterion_CE(predictions, labels.squeeze().cuda().long())
            # loss = criterion_SD(predictions_softmax, labels_onehot)
            loss.backward()
            optimizer.step()

        
            # iteration += 1
            if iteration%PRINT_AFTER_ITERATIONS == 0:

                # print('Epoch: {0}, Iteration: {1}, Loss: {2}, Valid dice score: {3}'.format(epoch, iteration, loss, score))
                print('Epoch: {0}, Iteration: {1}, Loss: {2}'.format(epoch, iteration, loss))

                image_args = {'normalize':True, 'range':(0, 1)}
                # viz.show_image_grid(images=images.cpu()[:, 0, ].unsqueeze(1), name='Images_train', image_args=image_args)
                vis.show_image_grid(images=predictions_softmax.cpu()[:, 0, ].unsqueeze(1), name='Predictions_1', image_args=image_args)
                vis.show_image_grid(images=predictions_softmax.cpu()[:, 1, ].unsqueeze(1), name='Predictions_2', image_args=image_args)
                vis.show_image_grid(images=labels.cpu(), name='Ground truth')
                vis.show_value(value=loss.item(), name='Train_Loss', label='Loss', counter=epoch + (iteration/MAX_ITERATIONS))
                

            if iteration == MAX_ITERATIONS:
                break

        score = model.predict(giana_valid_loader, SCORE_TYPE, MAX_VALIDATION_ITERATIONS, vis)
        vis.show_value(value=np.asarray([score]), name='TestDiceScore', label='Dice', counter=epoch)
        print('\n--------------------------------------------------\nEpoch: {0}, Score: {1}, Loss: {2}\n--------------------------------------------------\n'.format(epoch, score, loss))

        # # Clear memory #
        # del images, labels
        # torch.cuda.empty_cache()
        # ################
        # score_type = 'dice'
        # score = model.predict(giana_valid_loader, score_type)
        # if score_type=='dice':
        #     print('Epoch: {0}, Iteration: {1}, Loss: {2}, Dice score:{3}'.format(epoch, iteration, loss, score))
        # elif score_type=='accuracy':
        #     print('Epoch: {0}, Iteration: {1}, Loss: {2}, Accuracy score:{3}'.format(epoch, iteration, loss, score))

        # torch.cuda.empty_cache()

        

    


    ## Sequential
    # idx = 0
    # t0 = timeit.default_timer()
    # for images, labels in giana_train_loader:
    #     print("\n{0}: {1}   {2}".format(idx+1, images.size(), labels.size()))
    #     images, labels = giana_transform.apply_transform([images, labels])
    #      #plot_image_label(np.rollaxis(images.numpy(), 1, 4), 
    #      #                np.rollaxis(labels.numpy(), 1, 4))
    #     idx += 1
    #     if idx == n_iter:
    #         break

    # t1 = timeit.default_timer()
    # duration = int(math.ceil(1000 * (t1 - t0)))
    # print('\n\n\t\tDuration (ms): {0}, Average duration: {1}'.format(duration, duration/4))

    # plt.show()




if __name__ == "__main__":
    image_gt_file_list_all = '/home/avemuri/Dropbox/Temp/image_gt_data_file_list_all_640x640_linux.csv'
    train(image_gt_file_list_all)






# LEGACY CODE
# t0 = timeit.default_timer()
#print("{0}: {1}   {2}".format(idx+1, images.size(), labels.size()))
# pvl.show_value(idx+1, name='index')
#plot_image_label(np.rollaxis(images.numpy(), 1, 4), 
#                np.rollaxis(labels.numpy(), 1, 4))
# progress_bar.update(idx)

# time.sleep(0.5)  #  Expected processing time for each mini-batch
# t1 = timeit.default_timer()
# duration = int(math.ceil(1000 * (t1-t0)))
# print('{0}: Duration (ms): {1}'.format(idx+1, duration))




#t1 = timeit.default_timer()
#duration = int(math.ceil(1000 * (t1 - t0)))
#print('\n\n#######################################################################\n')
#print('Duration (sec): {0}, Average duration: {1}'.format(duration/1000, duration/(1000*n_iter)))
#print('\n#######################################################################\n\n\n')
    