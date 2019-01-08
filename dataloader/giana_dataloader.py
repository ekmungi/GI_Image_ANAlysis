
# import matplotlib.pyplot as plt

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchsample
from torchsample import CSVDataset
from torchsample.transforms import Compose, ChannelsFirst, TypeCast, ResizePadArray, AddChannel, RangeNormalize

from imgaug import augmenters as iaa
from torchsample.transforms import ImgAugTranform, ImgAugCoTranform




BATCH_SIZE = 6
IMAGE_SIZE = 512
TRAIN_RATIO = 0.8


def plot_image_label(images, labels):
    
    # fig = plt.figure(figsize=(15, 7))

    count = 0
    for image, label in zip(images, labels):

        fig = plt.figure(figsize=(5, 5))
        
        timer = fig.canvas.new_timer(interval = 1000)
        timer.add_callback(close_event)

        label = label[:,:,0]
        max_val = np.max([np.max(np.max(label)), 1.0])
        color_delta = [100, -80, -80]
        image[:, :, 0] = np.clip(image[:, :, 0] + color_delta[0] * (label/max_val), 0, 255)
        image[:, :, 1] = np.clip(image[:, :, 1] + color_delta[1] * (label/max_val), 0, 255)
        image[:, :, 2] = np.clip(image[:, :, 2] + color_delta[2] * (label/max_val), 0, 255)

        #print("\nImage ==> Max:{0}, Min:{1}".format(np.max(image), np.min(image)))
        #print("Label ==> Max:{0}, Min:{1}\n".format(np.max(label), np.min(label)))
        ax = plt.subplot(2, 1, 1)
        max_val = np.max(image)
        plt.imshow((image*255.0/max_val).astype(np.uint8))

        ax = plt.subplot(2, 1, 2)
        plt.imshow(label, cmap='gray')

        plt.tight_layout()
        ax.set_title('Sample #{}'.format(count+1))
        ax.axis('off')
        
        count += 1
        timer.start()
        plt.show()
    return None

def close_event():
    plt.close()

class GianaImgAugTransform(object):
    def __init__(self, image_transform_list, label_transform_list=None, joint_transform_list=None):
        self.image_transform = ImgAugTranform(image_transform_list)
        self.label_transform = ImgAugTranform(label_transform_list)
        self.joint_transform = ImgAugCoTranform(joint_transform_list)
        self.count = 0

    def apply_transform(self, inputs):

        images, labels = inputs[0], inputs[1]
        images= self.image_transform(images)
        if labels is not None:
            labels = self.label_transform(labels)
            images, labels = self.joint_transform(images, labels)
        self.count += 1

        return (images, labels)

def giana_data_pipeline(file_path):

    ## For loading complete tensor data into memory
    # image_data_file = 'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_data_all_640x640.npy'
    # label_data_file = 'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/gt_data_all_640x640.npy'

    ## For using csv file to read image locations and use them to load images at run time



    ## Transformations ############################################################
    ## TorchVision for getting the right format
    ## These transforms are applied by pytorch dataloader directly
    image_transform = transforms.Compose([
                                ResizePadArray(IMAGE_SIZE),
                                ChannelsFirst(),
                                TypeCast('float')
                                ])
    label_transform = transforms.Compose([
                                    ResizePadArray(IMAGE_SIZE),
                                    RangeNormalize(min_val=0, max_val=1),
                                    AddChannel(axis=2),
                                    ChannelsFirst()
                                ])
    joint_transform = None

    ## Imgaug tranforms to get the proper data augmentation
    add_image_transform_list = [
                                iaa.Sometimes(0.5, [iaa.GaussianBlur(sigma=(0.5, 2.0)),
                                                    iaa.Multiply((0.8, 1.2)),
                                                    iaa.ContrastNormalization((0.75, 1.5)),
                                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), 
                                                                                per_channel=True),
                                                    iaa.MotionBlur(size=(5, 10))])
                                ]

    add_joint_transform_list = [
                                # iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.2, 0.2), 
                                                                                    # "y": (-0.2, 0.2)},
                                                                # rotate=(-15, 15))),
                                iaa.Fliplr(0.5)
                                ]

    ##################################################################################

    file_list_df = pd.read_csv(file_path, header=None)

    giana_dataset = CSVDataset(file_list_df, input_transform=image_transform,
                                target_transform=label_transform, co_transform=joint_transform)

    
    train_dataset, valid_dataset = giana_dataset.train_test_split(TRAIN_RATIO)



    ## **Inefficient** XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ## When loading all the data into memory 
    # image_data = np.load(image_data_file)
    # label_data = np.load(label_data_file)
    # label_data = np.expand_dims(label_data, axis=3)

    # giana_dataset = TensorDataset(image_data, label_data, input_transform=image_transform,
    #                                 target_transform=label_transform, co_transform=joint_transform)

    ## XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    giana_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    giana_valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    giana_imgaug_transform = GianaImgAugTransform(image_transform_list=add_image_transform_list,
                                                  joint_transform_list=add_joint_transform_list)

    return (giana_imgaug_transform, giana_train_loader, giana_valid_loader)

