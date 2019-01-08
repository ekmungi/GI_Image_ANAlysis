import os
import pickle
from collections import defaultdict

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meddec.evaluation.evaluator import Evaluator, aggregate_scores
from meddec.expert_networks.Task07_Pancreas.datasets.Pancreas import PancreasDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment
from meddec.expert_networks.Task07_Pancreas.networks.UNET import UNet
from meddec.loss_functions.dice_loss import SoftDiceLoss


class PancreasExperiment(PytorchExperiment):
    def setup(self):
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        fold = 0
        tr_val_keys = splits[fold]['train']
        tr_keys = tr_val_keys[:-10]
        val_keys = tr_val_keys[-10:]
        test_keys = splits[fold]['val']

        base_dir = self.config.data_dir
        self.train_data_loader = PancreasDataSet(base_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size, keys=tr_keys, additional_slices=self.config.additional_slices)
        self.val_data_loader = PancreasDataSet(base_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size, keys=val_keys, mode="val", do_reshuffle=False, additional_slices=self.config.additional_slices)
        self.test_data_loader = PancreasDataSet(base_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size, keys=test_keys, mode="val", do_reshuffle=False, additional_slices=self.config.additional_slices)
        self.model = UNet(num_classes=3, in_channels=self.config.additional_slices*2 + 1, do_batchnorm=self.config.do_batchnorm)

        weighting = torch.tensor([1., 1., 1. ])
        if self.config.do_ce_weighting:
            # weighting = torch.tensor([1-0.9978, 1-0.0019, 1-0.0003])*10
            weighting = torch.tensor([1., 1., 20.])
        # self.loss = SoftDiceLoss(batch_dice=True)  # Softmax Layer im Unet einfügen!!!
        self.loss = torch.nn.CrossEntropyLoss(weight=weighting.cuda())  # Softmax Layer im Unet auskommentieren!!!

        if self.config.use_cuda:
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # self.load_checkpoint(name='/media/kleina/Data2/output/meddec/20180626-174058_pancreasexperiment/checkpoint/checkpoint_current.pth.tar',
        #                      save_types=("model"))
        self.save_checkpoint(name="checkpoint_start")
        # self.vlog.plot_model_structure(self.model,
        #                                [self.config.batch_size, 1, 28, 28],
        #                                name='Model Structure')
        self.elog.print('Experiment set up.')
        self.batch_counter = 0

    def train(self, epoch):
        self.elog.print('TRAIN')
        self.model.train()

        counter = 0
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            data = data_batch['data'][0][:,0,].float().cuda()
            target = data_batch['seg'][0][:, :, self.config.additional_slices, ].long().cuda()

            pred = self.model(data)
            loss = self.loss(pred, target.squeeze())
            loss.backward()
            self.optimizer.step()

            if (counter % 10) == 0:
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss',   label='Loss', counter=epoch + (counter / self.train_data_loader.data_loader.num_batches))

                # self.vlog.show_value(value=loss.item(), name='Loss', tag='Train Loss')
                self.clog.show_image_grid(data[:, self.config.additional_slices:self.config.additional_slices+1, ].float(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                # self.vlog.show_image_grid(data.float(), name="data_grid", title="Data Grid")
                self.clog.show_image_grid(target.float(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax", title="Unet", n_iter=epoch)
                self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(pred.data.cpu()[:, 2:3, ], name="unt2", normalize=True, scale_each=True, n_iter=epoch)

            counter += 1

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0][:, 0, ].float().cuda()
                target = data_batch['seg'][0][:, :, self.config.additional_slices, ].long().cuda()

                pred = self.model(data)
                loss = self.loss(pred, target.squeeze())

                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', label='Loss', counter=epoch+1)
        # self.vlog.show_value(value=np.mean(loss_list), name='Loss', tag='Val Loss')
        self.clog.show_image_grid(data[:, self.config.additional_slices:self.config.additional_slices+1, ].float(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        # self.vlog.show_image_grid(data.float(), name="data_grid", title="Data Grid")
        self.clog.show_image_grid(target.float(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 2:3, ], name="unt2_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        self.elog.print("Testing....")

        pred_dict = defaultdict(list)
        gt_dict = defaultdict(list)

        with torch.no_grad():
            for data_batch in self.test_data_loader:
                data = data_batch['data'][0][:, 0, ].float().cuda()
                target = data_batch['seg'][0][:, :, self.config.additional_slices, ].long().cuda()
                fnames = data_batch['fnames']

                pred = self.model(data)
                pred_label = torch.argmax(pred, dim=1, keepdim=True)

                for i, fname in enumerate(fnames):

                    pred_dict[fname[0]].append(pred_label[i].detach().cpu().numpy())
                    gt_dict[fname[0]].append(target[i].detach().cpu().numpy())

        test_ref_list = []
        for k in pred_dict.keys():
            test_ref_list.append((np.stack(pred_dict[k]), np.stack(gt_dict[k])))

        scores = aggregate_scores(test_ref_list, evaluator=Evaluator, json_author='AD', json_task='Task07_Pancreas', json_name='Task07_' + self.config.name,
                                  json_output_file=self.elog.work_dir + "/AD___" + self.config.name + '.json')

        self.add_result(value=scores['mean'][0]['Dice'], name="Test-Dice0")
        self.add_result(value=scores['mean'][1]['Dice'], name="Test-Dice1")
        self.add_result(value=scores['mean'][2]['Dice'], name="Test-Dice2")

        print("Scores:\n", scores)
