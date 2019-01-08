import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# from batchgenerators.transforms.spatial_transforms import MirrorTransform
# from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
# from batchgenerators.transforms.abstract_transforms import Compose
# from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
# from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, RicianNoiseTransform
# from batchgenerators.generators.data_generator_base import BatchGeneratorBase
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
# from batchgenerators.transforms.abstract_transforms import RndTransform

# from trixi.logger import PytorchVisdomLogger as pvl

import torchsample
from torchsample import TensorDataset, CSVDataset
from torchsample.transforms import Compose, Brightness, Contrast, ToTensor, TypeCast
from torchsample.transforms import RangeNormalize, RandomSaturation, ChannelsFirst, RandomGamma
from torchsample.transforms import RandomRotate, RandomFlip, AffineCompose
from torchsample.transforms import ResizePad, SpecialCrop, StdNormalize, Rotate, Translate


import Augmentor as augmentor

# mylog = pvl(name='GIANA')


def plot_image_label(images, labels):
    
    index = 0
    for image, label in zip(images, labels):
        print(label.shape)
        label = label[:,:,0]
        max_val = np.max([np.max(np.max(label)), 1.0])
        color_delta = [100/255.0, -80/255.0, -80/255.0]
        image[:, :, 0] = np.clip(image[:, :, 0] + color_delta[0] * (label/max_val), 0, 1)
        image[:, :, 1] = np.clip(image[:, :, 1] + color_delta[1] * (label/max_val), 0, 1)
        image[:, :, 2] = np.clip(image[:, :, 2] + color_delta[2] * (label/max_val), 0, 1)

        print("Max:{0}, Min:{1}".format(np.max(image), np.min(image)))
        ax = plt.subplot(1, 4, index + 1)
        plt.imshow(image)

        plt.tight_layout()
        ax.set_title('Sample #{}'.format(index))
        ax.axis('off')
        index += 1

def close_event():
    plt.close()

def main():
    ## For loading complete tensor data into memory
    # image_data_file = 'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_data_all_640x640.npy'
    # label_data_file = 'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/gt_data_all_640x640.npy'

    ## For using csv file to read image locations and use them to load images at run time
    image_gt_file_list_all = 'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_gt_data_file_list_all_640x640.csv'

    
    ## Transforms to apply
    image_transform = transforms.Compose([ResizePad(640, 'RGB'),
                                TypeCast('float'),
                                # RangeNormalize(0, 1),
                                # StdNormalize(),
                                RandomGamma(0.2, 1.0),
                                Brightness(0.1),
                                RandomSaturation(0.1, 0.2)
                                ])
    label_transform = transforms.Compose([ResizePad(640, 'L'), TypeCast('float')
                                ])
    joint_transform = Compose([RandomFlip(h=True, v=True)
                                #AffineCompose([Rotate(10), Translate((0.2, 0.2))]),
                                #SpecialCrop((400, 400))
                                ])

    image_gt_file_list_all_df = pd.read_csv(image_gt_file_list_all, header=None)

    ## Create Dataset object to read from CSV file
    giana_dataset = CSVDataset(image_gt_file_list_all_df, input_transform=image_transform,
                                target_transform=label_transform, co_transform=joint_transform)


    ## When loading all the data into memory
    # image_data = np.load(image_data_file)
    # label_data = np.load(label_data_file)
    # label_data = np.expand_dims(label_data, axis=3)

    # giana_dataset = TensorDataset(image_data, label_data, input_transform=image_transform,
    #                                 target_transform=label_transform, co_transform=joint_transform)
    
    ## Create pytorch dataloader
    giana_dataloader = DataLoader(giana_dataset, batch_size=4, shuffle=True)


    idx = 0
    for images, labels in giana_dataloader:
        print(images.size(), labels.size())
        fig = plt.figure(figsize=(10, 5))
        timer = fig.canvas.new_timer(interval = 2000)
        timer.add_callback(close_event)
        # idx = 0
        # for image, label in zip(images, labels):
        #     # plt.imshow(np.rollaxis(image.numpy(), 0, 3))
        plot_image_label(np.rollaxis(images.numpy(), 1, 4), 
                        np.rollaxis(labels.numpy(), 1, 4))
        
        timer.start()
        plt.show()

        idx += 1
        if idx == 4:
            break
        

    # plt.show()

if __name__ == "__main__":
    main()
