from argparse import ArgumentParser

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from segmentation_baseline.models import MulticlassModel, BinaryModel
from segmentation_baseline.datamodules import DataModule
from segmentation_baseline.utils import instantiate_from_config

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/base_config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(config))
    
    seed_everything(config['common']['seed'], workers=True)

    datamodule = DataModule(config)
    model = BinaryModel(config) if config['common']['task'] == 'binary' else MulticlassModel(config) 

    trainer = Trainer(
        max_epochs=config['common']['epochs'], 
        gpus=config['common']['gpus'],
        # logger=[instantiate_from_config(x) for x in config['loggers']],
        )

    trainer.fit(model, datamodule) 