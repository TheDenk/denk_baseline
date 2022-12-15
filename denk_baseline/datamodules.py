import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Sampler, SequentialSampler, DataLoader

from .utils import instantiate_from_config, get_obj_from_str


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train = instantiate_from_config(config['datasets']['train'])
        if self.train.augs: self.train.augs = get_obj_from_str(self.train.augs)()

        self.valid = instantiate_from_config(config['datasets']['valid'])
        if self.valid.augs: self.valid.augs = get_obj_from_str(self.valid.augs)()

    def train_dataloader(self):
        train_params = self.config['dataloaders']['train']['params']
        return DataLoader(self.train, 
                          sampler=RSNABalanceSampler(self.train, ratio=32),
                          batch_size=train_params.get('batch_size', 1),  
                          num_workers=train_params.get('num_workers', 1),
                          drop_last=train_params.get('drop_last', False),
                          pin_memory=train_params.get('pin_memory', True),
                          shuffle=train_params.get('shuffle', True),)

    def val_dataloader(self):
        valid_params = self.config['dataloaders']['train']['params']
        return DataLoader(self.valid, 
                          sampler=SequentialSampler(self.valid),
                          batch_size=valid_params.get('batch_size', 1),  
                          num_workers=valid_params.get('num_workers', 1),
                          drop_last=valid_params.get('drop_last', False),
                          pin_memory=valid_params.get('pin_memory', True),
                          shuffle=valid_params.get('shuffle', False),)


class RSNABalanceSampler(Sampler):
    def __init__(self, dataset, ratio=8, shuffle=True):
        self.r = ratio - 1
        self.dataset = dataset
        self.shuffle = shuffle
        self.pos_index = np.where(dataset.df.cancer>0)[0]
        self.neg_index = np.where(dataset.df.cancer==0)[0]

        self.length = self.r*int(np.floor(len(self.neg_index)/self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()

        if self.shuffle:
            np.random.shuffle(pos_index)
            np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.length//self.r).reshape(-1,1)

        index = np.concatenate([pos_index, neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length