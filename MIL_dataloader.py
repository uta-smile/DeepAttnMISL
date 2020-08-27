"""
Define pytorch dataloader for DeepAttnMISL


"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.model_selection import train_test_split

class MIL_dataloader():
    def __init__(self, data_path, cluster_num=10, train=True):

        if train:
            X_train, X_test = train_test_split(data_path, test_size=0.1, random_state=66)  # 10% validation

            traindataset = MIL_dataset(list_path=X_train, cluster_num = cluster_num, train=train,
                              transform=transforms.Compose([ToTensor()]))

            traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=4)

            valdataset = MIL_dataset(list_path=X_test, train=False, cluster_num=cluster_num,
                                       transform=transforms.Compose([ToTensor()]))

            valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = [traindataloader, valdataloader]

        else:
            testdataset = MIL_dataset(list_path=data_path, cluster_num = cluster_num, train=False,
                              transform=transforms.Compose([ToTensor()]))
            testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = testloader

    def get_loader(self):
        return self.dataloader


class MIL_dataset(Dataset):
    def __init__(self, list_path, cluster_num,  transform=None, train=True):
        """
        Give npz file path
        :param list_path:
        """

        self.list_path = list_path
        self.random = train
        self.transform = transform
        self.cluster_num = cluster_num

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):

        img_path = self.list_path[idx]

        Batch_set = []
        surv_time_train = []
        status_train = []

        all_vgg = []

        vgg_clus = [[] for i in range(self.cluster_num)]

        Train_vgg_file = np.load(img_path)

        con_vgg, con_path, con_cluster = [], [], []

        mask = np.ones(self.cluster_num, dtype=np.float32)

        for i in range(1):  # How many wsi in the patient


            cur_vgg = Train_vgg_file['vgg_features']
            cur_patient = Train_vgg_file['pid']
            cur_time = Train_vgg_file['time']
            cur_status = Train_vgg_file['status']
            cur_path = Train_vgg_file['img_path']
            cur_cluster = Train_vgg_file['cluster_num']

            for id, each_patch_cls in enumerate(cur_cluster):
                    vgg_clus[each_patch_cls].append(cur_vgg[id])

            Batch_set.append((cur_vgg, cur_patient, cur_status, cur_time, cur_cluster))

            np_vgg_fea = []
            for i in range(self.cluster_num):
                if len(vgg_clus[i]) == 0:
                    clus_feat = np.zeros((1, 4096), dtype=np.float32)
                    mask[i] = 0
                else:
                    if self.random:
                        curr_feat = vgg_clus[i]
                        ind = np.arange(len(curr_feat))
                        np.random.shuffle(ind)
                        clus_feat = np.asarray([curr_feat[i] for i in ind])
                    else:
                        clus_feat = np.asarray(vgg_clus[i])
                clus_feat = np.swapaxes(clus_feat, 1, 0)
                # clus_feat = np.expand_dims(clus_feat, 0)
                clus_feat = np.expand_dims(clus_feat, 1)
                np_vgg_fea.append(clus_feat)

            all_vgg.append(np_vgg_fea)

        for each_set in Batch_set:
            surv_time_train.append(each_set[3])
            status_train.append(each_set[2])

        surv_time_train = np.asarray(surv_time_train)
        status_train = np.asarray(status_train)

        np_cls_num = np.asarray(cur_cluster)

        sample = {'feat': all_vgg[0], 'mask':mask, 'time': surv_time_train[0], 'status':status_train[0], 'cluster_num': np_cls_num}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cluster_num = 10
        image, time, status = sample['feat'], sample['time'], sample['status']

        return {'feat': [torch.from_numpy(image[i]) for i in range(cluster_num)], 'time': torch.FloatTensor([time]), 'status':torch.FloatTensor([status]),
                'mask': torch.from_numpy(sample['mask']),
                'cluster_num': torch.from_numpy(sample['cluster_num'])
                }