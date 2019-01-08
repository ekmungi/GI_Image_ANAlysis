import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import cv2


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, RicianNoiseTransform
from batchgenerators.generators.data_generator_base import BatchGeneratorBase
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import RndTransform

# from trixi.logger import PytorchVisdomLogger as pvl

import torchsample
from torchsample import TensorDataset
from torchsample.transforms import Compose, Brightness, Contrast, ToTensor, TypeCast
from torchsample.transforms import RangeNormalize, RandomSaturation, ChannelsFirst, RandomGamma

# mylog = pvl(name='GIANA')


def plot_image_label(image, label, index):
    
    max_val = np.max([np.max(np.max(label)), 1.0])
    color_delta = [100, -20, -20]
    image[:, :, 0] = np.clip(image[:, :, 0] + color_delta[0] * (label/max_val), 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] + color_delta[1] * (label/max_val), 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] + color_delta[2] * (label/max_val), 0, 255)

    # print(np.max(np.max(image)))
    # print(image.shape, label.shape)
    ax = plt.subplot(2, 4, index + 1)
    plt.imshow(image.astype(np.uint8))
    # plt.imshow(image)
    ax = plt.subplot(2, 4, index + 5)
    plt.imshow(label, cmap='gray')

    plt.tight_layout()
    ax.set_title('Sample #{}'.format(index))
    ax.axis('off')

class GianaDataset(Dataset):
    """GIANA polyp dataset."""

    def __init__(self, image_data_file_list, label_data_file_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_data = []
        self.label_data = []
        for image_file, label_file in zip(image_data_file_list, label_data_file_list):
            self.image_data.append(np.load(image_file))
            self.label_data.append(np.load(label_file))
        
        print('Data loaded...')
        self.transform = transform
        self.selection_iter = cycle(list(range(len(self.image_data))))
        self.select_dataset()

    def __len__(self):
        return len(self.image_data[self.selected])

    def __getitem__(self, idx):
        image = np.array(self.image_data[self.selected][idx], dtype=np.float32)
        image = np.rollaxis(image, 2, 0)
        label = np.array(self.label_data[self.selected][idx], dtype=np.long)
        label = np.expand_dims(label, axis=0)
    
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        ## For torchsample lib
        # return (image, label)

        ## For batchgenerators lib
        return {'data':image, 'seg':label}

    def select_dataset(self):
        self.selected = next(self.selection_iter)

    def random_select_samples(self, batch_size, replace=False):
        selected_indices = np.random.randint(0, len(self.image_data[self.selected]), batch_size)
        data_dict = {'data':np.rollaxis(self.image_data[self.selected][selected_indices], 3, 1),
                        'seg':np.expand_dims(self.label_data[self.selected][selected_indices], axis=1)}
        return data_dict


class GianaDataGenerator(BatchGeneratorBase):

    def __init__(self, dataset, BATCH_SIZE, num_batches=None, seed=False):
        BatchGeneratorBase.__init__(self, dataset, BATCH_SIZE, num_batches, seed)
        self.dataset = dataset
        #np.random.shuffle(self.samples_list)
        #self.batch_index_iter = iter(self.samples_list)

    def __next__(self):
        if not self._iter_initialized:
            self._initialize_iter()
        if self._batches_generated >= self._num_batches:
            self._iter_initialized = False
            raise StopIteration
        minibatch = self.generate_train_batch()
        self._batches_generated += 1
        return minibatch

    def generate_train_batch(self, replace=False):
        self.dataset.select_dataset() # Selects the dataset to generate the batch from
        return self.dataset.random_select_samples(self.BATCH_SIZE, replace)



        



def main():
    image_data_file = ['C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_data_1.npy',
                        'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_data_2.npy',
                        'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_data_3.npy']
    label_data_file = ['C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/gt_data_1.npy',
                        'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/gt_data_2.npy',
                        'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/gt_data_3.npy']

    ## Using batchgenerators
    transform_list = []

    ## Spatial transform
    # transform_list.append(SpatialTransform((600, 600), np.array((600, 600)) // 2, 
    #                         do_elastic_deform=False, alpha=(0., 1500.), sigma=(30., 50.),
    #                         do_rotation=True, angle_z=(0, 2 * np.pi),
    #                         do_scale=True, scale=(0.3, 3.), 
    #                         border_mode_data='constant', border_cval_data=0, order_data=1,
    #                         random_crop=False))
    # transform_list.append(Mirror(axes=(2, 3)))

    ## Noise transforms
    # transform_list.append(GaussianNoiseTransform(noise_variance=(0.1, 0.5)))
    # transform_list.append(RicianNoiseTransform(noise_variance=(0, 0.3)))

    ## Color transforms
    transform_list.append(ContrastAugmentationTransform((0.3, 0.5), preserve_range=True))
    


    transformations = Compose(transform_list)
    
    giana_dataset = GianaDataset(image_data_file, label_data_file)
    giana_dataloader = GianaDataGenerator(giana_dataset, 4, 4)
    multithreaded_generator = MultiThreadedAugmenter(giana_dataloader, transformations, 4, 2, seeds=None)
    # 
    #for data_dict in multithreaded_generator:
    
    #data_dict = transformations(**data_dict)


    ## Using pytorch build-in transformations
    # transformations = transforms.Compose([#transforms.ToPILImage(),
    #                                         # transforms.RandomApply([transforms.RandomVerticalFlip(),
    #                                                                 # transforms.RandomHorizontalFlip()]),
    #                                         #transforms.ToTensor(),
    #                                         #Permute(),
    #                                         RicianNoiseTransform(noise_variance=(0, 200))
    #                                         #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                 # std=[0.229, 0.224, 0.225])
    #                                                                 ])
    # giana_dataset = GianaDataset(image_data_file, label_data_file)#, transform=transformations)
    # giana_dataloader = DataLoader(giana_dataset, batch_size=4, shuffle=True)

    ## Using torchsample
    # transformations = Compose([ToTensor(), 
    #                      TypeCast('float'), 
    #                      #ChannelsFirst(),
    #                      RangeNormalize(0,1),
    #                      RandomGamma(0.2,1.8),
    #                      Brightness(0.4),
    #                      RandomSaturation(0.5,0.9)
    #                     ])
    
    # giana_dataset = GianaDataset(image_data_file, label_data_file, transform=transformations)
    # giana_dataloader = DataLoader(giana_dataset, batch_size=4, shuffle=True)
    
    
    


    # for data_dict in giana_dataloader.next():
    #     print(data_dict)
    #     break
    for i in range(4):
        data_dict = next(giana_dataloader)
        data_dict = transformations(**data_dict)
        print("Dataset selected: {0}".format(giana_dataloader.dataset.selected))
        images, labels = data_dict['data'], data_dict['seg']
        plt.figure()
        idx = 0
        for image, label in zip(images, labels):
            # mylog.show_image(image, name="img plot"+str(idx), title="image title"+str(idx))
            # mylog.show_image(label, name="label plot"+str(idx), title="label title"+str(idx))
            # print(image.shape, label.shape)
            plot_image_label(np.rollaxis(image, 0, 3), np.squeeze(label, axis=0), idx)
            # print(idx, image.shape, label.shape, type(image))
            idx += 1
        # plt.show()
        # break
    plt.show()

    # for i in range(4):
    #     giana_dataset.select_dataset()
    #     print("Dataset selected: {0}".format(giana_dataset.selected))
    #     selected_idx = np.random.randint(len(giana_dataset))
    #     data_dict = giana_dataset[selected_idx]
    #     image, label = data_dict['data'], data_dict['seg']
    #     plot_image_label(np.rollaxis(image, 0, 3), np.squeeze(label, axis=0), i)

    # plt.show()

if __name__ == "__main__":
    main()
