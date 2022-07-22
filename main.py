from argparse import ArgumentParser

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from model import LightningModel
from datamodules import DataModule

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/base_config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = OmegaConf.load('./configs/base_config.yaml')

    seed_everything(config['common']['seed'], workers=True)

    datamodule = DataModule(config)
    model = LightningModel(config)

    trainer = Trainer(max_epochs=config['common']['epochs'], gpus=config['common']['gpus'])
    trainer.fit(model, datamodule) 