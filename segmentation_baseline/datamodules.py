import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .augs import get_train_augs, get_valid_augs
from .utils import instantiate_from_config

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train = instantiate_from_config(config['datasets']['train'])
        self.train.augs = get_train_augs()

        self.valid = instantiate_from_config(config['datasets']['valid'])
        self.valid.augs = get_valid_augs()

    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.config['common'].get('batch_size', 1),  
                          num_workers=self.config['common'].get('num_workers', 1),
                          drop_last=self.config['common'].get('drop_last', True),
                          pin_memory=self.config['common'].get('pin_memory', True),
                          shuffle=self.config['common'].get('shuffle', True),)

    def val_dataloader(self):
        return DataLoader(self.valid, 
                          batch_size=self.config['common'].get('batch_size', 1),  
                          num_workers=self.config['common'].get('num_workers', 1),
                          drop_last=self.config['common'].get('drop_last', False),
                          pin_memory=self.config['common'].get('pin_memory', True),
                          shuffle=self.config['common'].get('shuffle', False),)