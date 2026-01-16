import os
from argparse import ArgumentParser
from typing import Dict, Any

import torch
from pytorch_lightning import seed_everything

from denk_baseline.config import ExperimentConfig, load_config, save_config
from denk_baseline.loggers import LoggerFactory
from denk_baseline.datamodules import DataModule
from denk_baseline.model_utils import ModelExtractor
from denk_baseline.utils import get_obj_from_str


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    return vars(parser.parse_args())


def setup_experiment(config: Dict[str, Any]) -> ExperimentConfig:
    """Setup experiment configuration and save it."""
    exp_config = ExperimentConfig.from_dict(config)
    config.setdefault('general', {})['save_path'] = exp_config.save_path
    save_config(config, exp_config.save_path)
    return exp_config


def run_experiment(config: Dict[str, Any]) -> None:
    """Run the main experiment."""
    exp_config = setup_experiment(config)
    seed_everything(exp_config.seed, workers=True)

    # Initialize components
    datamodule = DataModule(config)
    model = get_obj_from_str(config['lightning_model'])(config)
    loggers = LoggerFactory.create_loggers(config)
    
    # Setup trainer
    trainer = get_obj_from_str(config['trainer']['target'])(
        logger=[v for _, v in loggers.items()] if loggers else None,
        **config['trainer']['params']
    )

    # Run training
    trainer.fit(model, datamodule)
    ModelExtractor.extract_checkpoints(model, exp_config.save_path)


def run_test(config: Dict[str, Any]) -> None:
    """Run model testing."""
    exp_config = setup_experiment(config)
    seed_everything(exp_config.seed, workers=True)

    datamodule = DataModule(config)
    model = get_obj_from_str(config['lightning_model'])(config)
    loggers = LoggerFactory.create_loggers(config)
    
    trainer = get_obj_from_str(config['trainer']['target'])(
        logger=[v for _, v in loggers.items()] if loggers else None,
        **config['trainer']['params']
    )
    
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args['config'])
    
    if args['test']:
        run_test(config)
    else:
        run_experiment(config)
