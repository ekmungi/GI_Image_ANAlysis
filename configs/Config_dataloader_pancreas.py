import torch
from torchvision import datasets, transforms
from trixi.util import Config

from datasets.Pancreas import PancreasDataLoader

from giana.dataloader.giana_dataloader import GianaDataset

def get_config():
    c = Config()

    c.batch_size = 6
    c.patch_size = 512
    c.n_epochs = 20
    c.learning_rate = 0.0002
    c.do_ce_weighting = True
    c.do_batchnorm = True
    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False
    c.rnd_seed = 1
    c.log_interval = 200
    c.base_dir='/media/kleina/Data2/output/meddec'
    c.data_dir='/media/kleina/Data2/Data/meddec/Task07_Pancreas_expert_preprocessed'
    c.split_dir='/media/kleina/Data2/Data/meddec/Task07_Pancreas_preprocessed'
    c.data_file = 'C:/dev/data/Endoviz2018/GIANA/polyp_detection_segmentation/image_gt_data_file_list_all_640x640.csv'
    c.additional_slices=5
    c.name=''

    print(c)
    return c
